from data_reader import game_info_detailed_2

result_0_avg = 0
result_0_counter = 0
result_1_avg = 0
result_1_counter = 0

data_idx = 5

for i in range(len(game_info_detailed_2)):
  if i > 50:
    if game_info_detailed_2[i][0] == 0:
      result_0_avg += game_info_detailed_2[i][data_idx]
      result_0_counter += 1
    else:
      result_1_avg += game_info_detailed_2[i][data_idx]
      result_1_counter += 1

result_0_avg /= result_0_counter
result_1_avg /= result_1_counter

print("Result 0 average: " + str(result_0_avg))
print("Result 1 average: " + str(result_1_avg))

print("Result 0 counter: " + str(result_0_counter))
print("Result 1 counter: " + str(result_1_counter))