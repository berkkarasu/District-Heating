import joblib
import numpy as np
from itertools import repeat
from operator import attrgetter
from deap import base, tools, creator, algorithms
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import config
import genetic

"""
def main():
  synth_batch_data = joblib.load('paper_synth_data.pk')
  total_num_run = 1
  res_dict_filename = "res_dict_5000_gen.pk"

  res_dict = dict()
  for k, v in synth_batch_data.items():
    if k == (3,12):
      num_radiators, application_period_in_hours = k
      current_temperature_list = v['init_temp']
      demanded_temperature_list = v['demanded_temp']
      
      boiler_inflection_point_list = genetic.get_boiler_inflection_points(
        num_radiators, 
        config.VALVE_MODULATION_ENERGY_CONSUMPTIONS[-1]
      )
      temperature_prediction_coef_list = genetic.get_temperature_prediction_coefficients(
        num_radiators
      )
      
      res_dict[(num_radiators, application_period_in_hours)] = {
        "initial_temperatures": current_temperature_list,
        "demanded_temperatures": demanded_temperature_list
      }
      
      print()
      print('#' * 100)
      print()
      print(f'Number of radiators: {num_radiators}')
      print(f'Application period in hours: {application_period_in_hours}')
      
      for num_run in range(1, total_num_run + 1):
        print()
        print(f'Run number: {num_run}')

        run_res_dict, _ = genetic.genetic_solver_run(
          num_radiators, 
          config.POPULATION_SIZE, 
          config.OFFSPRING_SIZE, 
          config.CROSSOVER_PROBA,
          config.CX_APARTMENT_PROBA,
          config.CX_TIME_POINT_PROBA,
          config.MUTATION_PROBA,
          config.MUTATION_APARTMENT_PROBA,
          config.MUTATION_TIME_POINT_PROBA,
          config.DEMAND_GRANULARITY, 
          config.SUPPLY_GRANULARITY, 
          application_period_in_hours, 
          config.DATA_UNIT_TIME, 
          config.GENERATION_LENGTH, 
          config.VALVE_MIN_MODULATION,
          config.VALVE_MAX_MODULATION, 
          current_temperature_list, 
          demanded_temperature_list, 
          config.VALVE_MODULATION_LIMITS, 
          config.VALVE_MODULATION_ENERGY_CONSUMPTIONS, 
          boiler_inflection_point_list, 
          config.BOILER_COSTS,
          temperature_prediction_coef_list
        )
        
        res_dict[(num_radiators, application_period_in_hours)][num_run] = run_res_dict
        joblib.dump(res_dict, res_dict_filename)
        
        print(
          "\tCurrent practice energy consumption:", 
          res_dict[k][num_run]["current_practice"]["energy_consumption_sum"]
        )
        print(
          "\tRecommended schedule energy consumption:", 
          res_dict[k][num_run]["recommended_schedule"]["energy_consumption_sum"]
        )
        print(
          "\tGA runtime:", 
          res_dict[k][num_run]["runtime"]
        )


  if __name__ == "__main__":
      main()


"""
def main():
  synth_batch_data = joblib.load('paper_synth_data.pk')
  total_num_run = 1
  res_dict_filename = "res_dict_5000_gen.pk"

  res_dict = dict()
  for k, v in synth_batch_data.items():
    num_radiators, application_period_in_hours = k
    current_temperature_list = v['init_temp']
    demanded_temperature_list = v['demanded_temp']
    
    boiler_inflection_point_list = genetic.get_boiler_inflection_points(
      num_radiators, 
      config.VALVE_MODULATION_ENERGY_CONSUMPTIONS[-1]
    )
    temperature_prediction_coef_list = genetic.get_temperature_prediction_coefficients(
      num_radiators
    )
    
    res_dict[(num_radiators, application_period_in_hours)] = {
      "initial_temperatures": current_temperature_list,
      "demanded_temperatures": demanded_temperature_list
    }
    
    print()
    print('#' * 100)
    print()
    print(f'Number of radiators: {num_radiators}')
    print(f'Application period in hours: {application_period_in_hours}')
    
    for num_run in range(1, total_num_run + 1):
      print()
      print(f'Run number: {num_run}')

      run_res_dict, _ = genetic.genetic_solver_run(
        num_radiators, 
        config.POPULATION_SIZE, 
        config.OFFSPRING_SIZE, 
        config.CROSSOVER_PROBA,
        config.CX_APARTMENT_PROBA,
        config.CX_TIME_POINT_PROBA,
        config.MUTATION_PROBA,
        config.MUTATION_APARTMENT_PROBA,
        config.MUTATION_TIME_POINT_PROBA,
        config.DEMAND_GRANULARITY, 
        config.SUPPLY_GRANULARITY, 
        application_period_in_hours, 
        config.DATA_UNIT_TIME, 
        config.GENERATION_LENGTH, 
        config.VALVE_MIN_MODULATION,
        config.VALVE_MAX_MODULATION, 
        current_temperature_list, 
        demanded_temperature_list, 
        config.VALVE_MODULATION_LIMITS, 
        config.VALVE_MODULATION_ENERGY_CONSUMPTIONS, 
        boiler_inflection_point_list, 
        config.BOILER_COSTS,
        temperature_prediction_coef_list
      )
      
      res_dict[(num_radiators, application_period_in_hours)][num_run] = run_res_dict
      joblib.dump(res_dict, res_dict_filename)
      
      print(
        "\tCurrent practice energy consumption:", 
        res_dict[k][num_run]["current_practice"]["energy_consumption_sum"]
      )
      print(
        "\tRecommended schedule energy consumption:", 
        res_dict[k][num_run]["recommended_schedule"]["energy_consumption_sum"]
      )
      print(
        "\tGA runtime:", 
        res_dict[k][num_run]["runtime"]
      )


if __name__ == "__main__":
    main()
