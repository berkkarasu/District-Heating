import time
import random
import numpy as np
from itertools import repeat
from operator import attrgetter
from deap import base, tools, creator, algorithms
from scoop import futures
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def get_boiler_inflection_points(num_radiators, max_energy_consumption_coefficient):
  boiler_inflection_point_list = [
    float(num_radiators * max_energy_consumption_coefficient * (1 / 6)),
    float(num_radiators * max_energy_consumption_coefficient * (1 / 2))
  ]
  return np.array(boiler_inflection_point_list)


def get_temperature_prediction_coefficients(num_radiators):
  temp_pred_coef_list = [[-.175469, 0, 15.095, 15.49728, 15.89319]] * num_radiators # 12.01728
  return np.array(temp_pred_coef_list).T


def calculate_delta_temp_list(
  curr_temp_list, 
  target_temp_list
):
  return target_temp_list - curr_temp_list


def set_modulation_list(
  curr_temp_list, 
  target_temp_list, 
  valve_modulation_limit_list
):
  delta_temp_list = calculate_delta_temp_list(curr_temp_list, target_temp_list)
  modulations = []
  for i in range(delta_temp_list.shape[0]):
    delta_temp = delta_temp_list[i]

    if delta_temp <= valve_modulation_limit_list[0]:
      modulations.append(0)
    elif valve_modulation_limit_list[0] <= delta_temp <= valve_modulation_limit_list[1]:
      modulations.append(1)
    elif valve_modulation_limit_list[1] <= delta_temp <= valve_modulation_limit_list[2]:
      modulations.append(2)
    elif valve_modulation_limit_list[2] <= delta_temp <= 10:
      modulations.append(3)
  return np.array(modulations)


def calculate_energy_consumption_list(
  valve_modulation_energy_consumption_list, 
  valve_modulation_list
):
  return valve_modulation_energy_consumption_list[valve_modulation_list]


def predict_temperature_with_regression(
  target_temp_list, 
  curr_temp_list, 
  demand_total_time, 
  supply_granularity, 
  valve_modulation_limit_list, 
  temperature_prediction_coef_list, 
  num_radiators, 
  valve_modulation_energy_consumption_list
):
  hum_pred = 44
  iter_curr_temp_list = curr_temp_list.copy()
  temperature_prediction_list = list()
  energy_consumption_list = list()
  for t in range(demand_total_time):
    target_temp_list_at_t = target_temp_list[:, int(t / supply_granularity)]
    
    if any([i not in [0, 1, 2, 3] for i in target_temp_list_at_t]):
      valve_modulation_list = set_modulation_list(
        iter_curr_temp_list, 
        target_temp_list_at_t, 
        valve_modulation_limit_list
      )
    else:
      valve_modulation_list = target_temp_list_at_t
    
    temp_prediction_coef_list_over_hum_pred = temperature_prediction_coef_list[1:5] / hum_pred
    temp_diff_pred = np.array(
      [
        temp_prediction_coef_list_over_hum_pred[valve_modulation_list[f], f] 
        for f in range(num_radiators)
      ]
    ) + temperature_prediction_coef_list[0] / hum_pred
    iter_curr_temp_list += temp_diff_pred
    temperature_prediction_list.append(iter_curr_temp_list.copy())
    valve_energy_consumption_list = calculate_energy_consumption_list(
      valve_modulation_energy_consumption_list, 
      valve_modulation_list
    )
    energy_consumption_list.append(valve_energy_consumption_list.copy())
  return np.array(temperature_prediction_list).T, np.array(energy_consumption_list).T


def generate_individual(
  supply_total_time
):
  ind = np.random.choice(
    [0, 1, 2, 3], 
    p = [0.4, 0.3, 0.2, 0.1],
    size=(supply_total_time, )
  )
  return ind


def crossover(
  ind1,
  ind2,
  cx_apartment_proba,
  cx_time_point_proba
):
  apartment_length, time_length = ind1.shape
  
  for i in range(apartment_length):
    if random.random() < cx_apartment_proba:
      if random.random() < 0.2:
        indices = random.sample(range(time_length), 2)
        t0 = min(indices)
        t1 = max(indices)

        ind1_val = ind1[i][t0 : t1]
        ind2_val = ind2[i][t0 : t1]

        ind1[i][t0 : t1] = ind2_val
        ind2[i][t0 : t1] = ind1_val
      else:
        for j in range(time_length):
          if random.random() < cx_time_point_proba:
            ind1_val = ind1[i][j]
            ind2_val = ind2[i][j]

            ind1[i][j] = ind2_val
            ind2[i][j] = ind1_val

  return ind1, ind2


def mutate(
  individual, 
  lower_limit, 
  upper_limit, 
  mutation_apartment_proba,
  mutation_time_point_proba
):
  apartment_length, time_length = individual.shape
    
  for i in range(apartment_length):
    if random.random() < mutation_apartment_proba:
      for j in range(time_length):
        if random.random() < mutation_time_point_proba:
          val = individual[i][j]
          
          if random.random() < 0.2:
            new_val = np.random.choice(
              [0, 1, 2, 3],
              p = [0.4, 0.3, 0.2, 0.1]
            )
          else:
            if val == lower_limit:
              new_val = lower_limit + 1
            elif val == upper_limit:
              new_val = upper_limit - 1
            else:
              new_val = val + np.random.choice([-1, 1])

          individual[i][j] = new_val
  return individual,


def selection(
  individuals, 
  k, 
  fit_attr="fitness"
):
  best_selection_alpha = 0.8
  sorted_individuals = sorted(individuals, key=attrgetter(fit_attr), reverse=True)

  selected_best = tools.selBest(
    individuals,
    int(k * best_selection_alpha), 
    fit_attr=fit_attr
  )
  selected_random = tools.selRandom(
    sorted_individuals[int(len(individuals) * best_selection_alpha) :], 
    int(k * (1 - best_selection_alpha))
  )
  return selected_best + selected_random


def calculate_unmet_demand(
  temperature_prediction_list, 
  demanded_temperature_list
):
  return temperature_prediction_list - demanded_temperature_list


def calculate_rmse(
  li
):
  return np.sqrt((li ** 2).mean())


def calculate_custom_score(
  li
):
  temp_li = np.abs(li[li < 0])
  return ((10 + 100 * temp_li) ** 2).sum()


def calculate_negative_unmet_demand(
  unmet_demand_list
):
  negative_unmet_demand_list = unmet_demand_list.copy()
  negative_unmet_demand_list[negative_unmet_demand_list > 0] = 0.0
  return negative_unmet_demand_list


def evaluate_unmet_demand_one_way(
  unmet_demand_list
):
  negative_unmet_demand_list = calculate_negative_unmet_demand(unmet_demand_list)
  return calculate_custom_score(negative_unmet_demand_list)


def evaluate_unmet_demand_two_way(
  unmet_demand_list
):
  return calculate_rmse(unmet_demand_list)


def evaluate_energy_consumption(
  energy_consumption_list, 
  boiler_inflection_point_list, 
  boiler_cost_list
):
  total_energy_consumption_list = energy_consumption_list.sum(axis=0)
  boiler_energy_consumption_cost_list = np.where(
    total_energy_consumption_list <= boiler_inflection_point_list[0], 
    boiler_cost_list[0],
    np.where(
      total_energy_consumption_list <= boiler_inflection_point_list[1], 
      boiler_cost_list[1], 
      boiler_cost_list[2]
    )
  )
  
#   boiler_energy_consumption_cost_list **= 10
  return boiler_energy_consumption_cost_list.sum()


def evaluate(
  target_temp_list, 
  curr_temp_list, 
  demand_total_time, 
  supply_granularity, 
  valve_modulation_limit_list, 
  temperature_prediction_coef_list, 
  num_radiators, 
  valve_modulation_energy_consumption_list, 
  demanded_temperature_list,
  boiler_inflection_point_list, 
  boiler_cost_list
):
  temperature_prediction_list, energy_consumption_list = predict_temperature_with_regression(
    target_temp_list, 
    curr_temp_list, 
    demand_total_time, 
    supply_granularity,
    valve_modulation_limit_list, 
    temperature_prediction_coef_list, 
    num_radiators,
    valve_modulation_energy_consumption_list
  )

  unmet_demand_list = calculate_unmet_demand(
    temperature_prediction_list, 
    demanded_temperature_list
  )
  demand_evaluation = evaluate_unmet_demand_one_way(unmet_demand_list)

  energy_consumption_evaluation = evaluate_energy_consumption(
    energy_consumption_list, 
    boiler_inflection_point_list, 
    boiler_cost_list
  )
  
  score = demand_evaluation + energy_consumption_evaluation

  return score,


creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)


def genetic_solver_run(
  num_radiators, 
  population_size, 
  offspring_size, 
  crossover_proba, 
  cx_apartment_proba,
  cx_time_point_proba,
  mutation_proba, 
  mutation_apartment_proba, 
  mutation_time_point_proba,
  demand_granularity, 
  supply_granularity, 
  application_period, 
  data_unit_time, 
  generation_length, 
  valve_min_modulation,
  valve_max_modulation, 
  current_temperature_list, 
  demanded_temperature_list, 
  valve_modulation_limit_list, 
  valve_modulation_energy_consumption_list, 
  boiler_inflection_point_list, 
  boiler_cost_list,
  temperature_prediction_coef_list
):
  res_dict = dict()
  
  demand_total_time = int(application_period * demand_granularity)
  supply_total_time = int(application_period * demand_granularity / supply_granularity)
  
  demanded_temperature_list = np.repeat(
    demanded_temperature_list, 
    demand_granularity
  ).reshape(num_radiators, -1)
  
  res_dict["demanded_temperature_list"] = demanded_temperature_list
  
  current_practice_schedule_list = demanded_temperature_list

  current_practice_schedule_evaluation = evaluate(
    current_practice_schedule_list, 
    curr_temp_list=current_temperature_list, 
    demand_total_time=demand_total_time,
    supply_granularity=supply_granularity, 
    valve_modulation_limit_list=valve_modulation_limit_list, 
    temperature_prediction_coef_list=temperature_prediction_coef_list, 
    num_radiators=num_radiators,
    valve_modulation_energy_consumption_list=valve_modulation_energy_consumption_list, 
    demanded_temperature_list=demanded_temperature_list,
    boiler_inflection_point_list=boiler_inflection_point_list, 
    boiler_cost_list=boiler_cost_list
  )
  (
    current_practice_temperature_prediction_list, 
    current_practice_energy_consumption_list
  ) = predict_temperature_with_regression(
    current_practice_schedule_list,
    current_temperature_list, 
    demand_total_time, 
    supply_granularity, 
    valve_modulation_limit_list, 
    temperature_prediction_coef_list,
    num_radiators, 
    valve_modulation_energy_consumption_list
  )
  current_practice_unmet_demand_list = calculate_unmet_demand(
    current_practice_temperature_prediction_list, 
    demanded_temperature_list
  )
  current_practice_total_energy_consumption_list = current_practice_energy_consumption_list.sum(axis=0)
  current_practice_boiler_energy_consumption_cost_list = np.where(
    current_practice_total_energy_consumption_list <= boiler_inflection_point_list[0],
    boiler_cost_list[0],
    np.where(
      current_practice_total_energy_consumption_list <= boiler_inflection_point_list[1], 
      boiler_cost_list[1], 
      boiler_cost_list[2]
    )
  )
  
  res_dict["current_practice"] = {
    "evaluation_result": current_practice_schedule_evaluation[0],
    "negative_unmet_demand_mean": calculate_negative_unmet_demand(current_practice_unmet_demand_list).mean(),
    "energy_consumption_sum": current_practice_boiler_energy_consumption_cost_list.sum(),
    "energy_consumption_mean": current_practice_boiler_energy_consumption_cost_list.mean(),
    "list": current_practice_schedule_list,
    "temperature_prediction_list": current_practice_temperature_prediction_list,
    "unmet_demand_list": current_practice_unmet_demand_list,
    "energy_consumption_list": current_practice_boiler_energy_consumption_cost_list
  }
  
  toolbox = base.Toolbox()
  toolbox.register("map", futures.map)
  toolbox.register(
    "generate_individual", 
    generate_individual, 
    supply_total_time=supply_total_time
  )
  toolbox.register(
    "mate", 
    crossover,
    cx_apartment_proba = cx_apartment_proba,
    cx_time_point_proba = cx_time_point_proba
  )
  toolbox.register(
    "mutate", 
    mutate, 
    lower_limit = valve_min_modulation, 
    upper_limit = valve_max_modulation,
    mutation_apartment_proba = mutation_apartment_proba,
    mutation_time_point_proba = mutation_time_point_proba
  )
  toolbox.register(
    "select", 
    selection
  )

  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)

  start_time = time.time()
  pop_list = []
  res_dict["internal_logs"] = []
  i = 0
  while i < num_radiators:
    if num_radiators - i > 4 or (num_radiators - i) % 3 == 0:
      num = 3
    else:
      num = 2
      
    print(f"Radiators {', '.join([str(j) for j in range(i + 1, i + num + 1)])}")
    toolbox.register(
      "individual", 
      tools.initRepeat,
      container=creator.Individual, 
      func=toolbox.generate_individual, 
      n=num
    )
    toolbox.register(
      "population", 
      tools.initRepeat, 
      container=list, 
      func=toolbox.individual
    )
    toolbox.register(
      "evaluate", 
      evaluate, 
      curr_temp_list=current_temperature_list[i : i + num], 
      demand_total_time=demand_total_time, 
      supply_granularity=supply_granularity,
      valve_modulation_limit_list=valve_modulation_limit_list, 
      temperature_prediction_coef_list=temperature_prediction_coef_list[:, i : i + num], 
      num_radiators=num,
      valve_modulation_energy_consumption_list=valve_modulation_energy_consumption_list, 
      demanded_temperature_list=demanded_temperature_list[i : i + num, :],
      boiler_inflection_point_list=boiler_inflection_point_list * num / num_radiators, 
      boiler_cost_list=boiler_cost_list
    )

    internal_pop = toolbox.population(n=population_size)
    
    internal_pop, internal_logs = algorithms.eaMuPlusLambda(
      internal_pop, 
      toolbox, 
      mu=population_size, 
      lambda_=offspring_size, 
      cxpb=crossover_proba, 
      mutpb=mutation_proba, 
      ngen=generation_length,
      stats=stats, 
      verbose=False
    )

    pop_list.append(internal_pop)
    res_dict["internal_logs"].append(internal_logs)

    i += num
    
  toolbox.register(
    "individual", 
    tools.initRepeat,
    container=creator.Individual, 
    func=toolbox.generate_individual, 
    n=num_radiators
  )
  toolbox.register(
    "population", 
    tools.initRepeat, 
    container=list, 
    func=toolbox.individual
  )
  toolbox.register(
    "evaluate", 
    evaluate, 
    curr_temp_list=current_temperature_list, 
    demand_total_time=demand_total_time, 
    supply_granularity=supply_granularity,
    valve_modulation_limit_list=valve_modulation_limit_list, 
    temperature_prediction_coef_list=temperature_prediction_coef_list, 
    num_radiators=num_radiators,
    valve_modulation_energy_consumption_list=valve_modulation_energy_consumption_list, 
    demanded_temperature_list=demanded_temperature_list,
    boiler_inflection_point_list=boiler_inflection_point_list, 
    boiler_cost_list=boiler_cost_list
  )
  
  pop = [creator.Individual(ind) for ind in np.concatenate(pop_list, axis=1)]
  
  # Add OR solution to the population
#   pop[0] = creator.Individual(or_set_modulation)

  hof = tools.HallOfFame(1, similar=np.array_equal)
  
  print(f"All radiators:")
  pop, log = algorithms.eaMuPlusLambda(
    pop, 
    toolbox, 
    mu=population_size, 
    lambda_=offspring_size, 
    cxpb=crossover_proba, 
    mutpb=mutation_proba, 
    ngen=generation_length,
    stats=stats, 
    halloffame=hof, 
    verbose=False
  )
  end_time = time.time()
  
  res_dict["final_log"] = log
  res_dict["runtime"] = end_time - start_time

  recommended_schedule_list = hof[0]
  
  recommended_schedule_evaluation = evaluate(
    recommended_schedule_list, 
    curr_temp_list=current_temperature_list, 
    demand_total_time=demand_total_time,
    supply_granularity=supply_granularity, 
    valve_modulation_limit_list=valve_modulation_limit_list, 
    temperature_prediction_coef_list=temperature_prediction_coef_list, 
    num_radiators=num_radiators,
    valve_modulation_energy_consumption_list=valve_modulation_energy_consumption_list, 
    demanded_temperature_list=demanded_temperature_list,
    boiler_inflection_point_list=boiler_inflection_point_list, 
    boiler_cost_list=boiler_cost_list
  )
  (
    recommended_schedule_temperature_prediction_list, 
    recommended_schedule_energy_consumption_list
  ) = predict_temperature_with_regression(
    recommended_schedule_list,
    current_temperature_list, 
    demand_total_time, 
    supply_granularity, 
    valve_modulation_limit_list, 
    temperature_prediction_coef_list,
    num_radiators, 
    valve_modulation_energy_consumption_list
  )
  recommended_schedule_unmet_demand_list = calculate_unmet_demand(
    recommended_schedule_temperature_prediction_list, 
    demanded_temperature_list
  )
  recommended_schedule_total_energy_consumption_list = recommended_schedule_energy_consumption_list.sum(axis=0)
  recommended_schedule_boiler_energy_consumption_cost_list = np.where(
    recommended_schedule_total_energy_consumption_list <= boiler_inflection_point_list[0],
    boiler_cost_list[0], 
    np.where(
      recommended_schedule_total_energy_consumption_list <= boiler_inflection_point_list[1], 
      boiler_cost_list[1],
      boiler_cost_list[2]
    )
  )
  
  res_dict["recommended_schedule"] = {
    "evaluation_result": recommended_schedule_evaluation[0],
    "negative_unmet_demand_mean": calculate_negative_unmet_demand(recommended_schedule_unmet_demand_list).mean(),
    "energy_consumption_sum": recommended_schedule_boiler_energy_consumption_cost_list.sum(),
    "energy_consumption_mean": recommended_schedule_boiler_energy_consumption_cost_list.mean(),
    "list": recommended_schedule_list,
    "temperature_prediction_list": recommended_schedule_temperature_prediction_list,
    "unmet_demand_list": recommended_schedule_unmet_demand_list,
    "energy_consumption_list": recommended_schedule_boiler_energy_consumption_cost_list
  }
  
  return res_dict, pop