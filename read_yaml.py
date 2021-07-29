# test the yaml !
import sys
import yaml

path2config = sys.argv[1]
#%% load config file


#with open("../config_tmp.yml", 'r') as stream:
with open(path2config, 'r') as stream:
    try:
        # print(yaml.safe_load(stream))

        config = yaml.safe_load(stream)

    except yaml.YAMLError as exc:
        print(exc)

#%% Go through levels of config file (headers), and save all values
## to the same list. Then, assign those values to standard variables
print(config)
# print(config['paths/key'])
#
variable_list = []
value_list = []


for headers in config.values():

    for value_dict in headers:
        print(value_dict)

        for k, v in value_dict.items():
            print(v,k)

            variable_list.append(k)
            value_list.append(v)



## assign values to variables
for index, value in enumerate(value_list):
    exec(f"{variable_list[index]} = value")
