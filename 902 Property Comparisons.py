import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Load datasets
"""
ground = pd.read_csv('Ground_Truth_properties.csv')
single = pd.read_csv('Single_Shot_properties.csv')
multiple = pd.read_csv('Multiple_Shots_properties.csv')

"""
Sum values in all columns in all datasets
"""
ground_area_sum = ground['area'].sum()  # /1000
ground_euler_sum = ground['euler_number'].sum()  # *100
ground_extent_sum = ground['extent'].sum()  # *100
ground_perimeter_sum = ground['perimeter'].sum()  # /10
ground_solidity_sum = ground['solidity'].sum()  # *100

single_area_sum = single['area'].sum()  # /1000
single_euler_sum = single['euler_number'].sum()
single_extent_sum = single['extent'].sum()
single_perimeter_sum = single['perimeter'].sum()  # /10
single_solidity_sum = single['solidity'].sum()

multiple_area_sum = multiple['area'].sum()  # /1000
multiple_euler_sum = multiple['euler_number'].sum()
multiple_extent_sum = multiple['extent'].sum()
multiple_perimeter_sum = multiple['perimeter'].sum()  # /10
multiple_solidity_sum = multiple['solidity'].sum()

"""
Create a dataframe and properly populate it
"""
master_data = pd.DataFrame(
    {
        'Parameters': ['area.10e-3', 'euler_number', 'extent', 'perimeter.10e-1', 'solidity',
                       'area.10e-3', 'euler_number', 'extent', 'perimeter.10e-1', 'solidity',
                       'area.10e-3', 'euler_number', 'extent', 'perimeter.10e-1', 'solidity'],

        'Values': [ground_area_sum, ground_euler_sum, ground_extent_sum, ground_perimeter_sum, ground_solidity_sum,
                   single_area_sum, single_euler_sum, single_extent_sum, single_perimeter_sum, single_solidity_sum,
                   multiple_area_sum, multiple_euler_sum, multiple_extent_sum, multiple_perimeter_sum,
                   multiple_solidity_sum],

        'Labels': ['GT.e100', 'GT.e100', 'GT.e100', 'GT.e100', 'GT.e100',
                   'SS', 'SS', 'SS', 'SS', 'SS',
                   'MS', 'MS', 'MS', 'MS', 'MS']
    }
)


"""
Design a grouped bar plot with your sns
"""
# set seaborn plotting aesthetics
sns.set(style='white')

# create grouped horizontal bar chart
# The axes (x and y attributes below)
# were  simply interchanged in order
# to get a horizontal bar chart
sns.barplot(y='Parameters',
            x='Values',
            hue='Labels',
            data=master_data,
            palette=['purple', 'steelblue', 'black'])

# add overall title
# plt.title('Customers by Time & Day of Week', fontsize=16)

# add axis titles
plt.xlabel('Properties')
plt.ylabel('Values')

# rotate x-axis labels
plt.xticks(rotation=0)

# plt.show()

"""
Show sum values
"""
grnd = [ground_area_sum, ground_euler_sum,
        ground_extent_sum, ground_perimeter_sum,
        ground_solidity_sum, ]
sngl = [single_area_sum, single_euler_sum,
        single_extent_sum, single_perimeter_sum,
        single_solidity_sum]

mltpl = [multiple_area_sum, multiple_euler_sum,
         multiple_extent_sum, multiple_perimeter_sum,
         multiple_solidity_sum]


def print_list(listt):
    for value in listt:
        c = round(value, 2)
        print(c)


print_list(sngl)
