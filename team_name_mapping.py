# Grok helped rewrite this code so it maps CHO and CHA to the same value
# Class to help with mapping string values to ints for one-hot encoding
class TeamNameMapping:
  def __init__(self, array):
    values = set(array)
    self._name_to_int_mapping = dict()
    current_index = 0

    # Process all team names, treating "CHO" and "CHA" as the same
    for team_name in values:
      if team_name in ["CHO", "CHA"]:
        # Assign the same index to both "CHO" and "CHA"
        if "CHO" not in self._name_to_int_mapping and "CHA" not in self._name_to_int_mapping:
          self._name_to_int_mapping["CHO"] = current_index
          self._name_to_int_mapping["CHA"] = current_index
          current_index += 1
      else:
        # Assign unique index to other team names
        self._name_to_int_mapping[team_name] = current_index
        current_index += 1

  def get_team_name_to_int_mapping(self):
    return self._name_to_int_mapping