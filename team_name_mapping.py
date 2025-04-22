# Class to help with mapping string values to ints for one-hot encoding
class TeamNameMapping:
  def __init__(self, array):
    values = set(array)
    self._name_to_int_mapping = {team_name: i for i, team_name in enumerate(values)}

  def get_team_name_to_int_mapping(self):
    return self._name_to_int_mapping