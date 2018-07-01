import numpy
import pandas

# lists of data
list_00 = [[1, 2, 3], [7, 8, 9]]
list_col = [4, 5]
list_row = [10, 11, 12]

# create arrays
array_00 = numpy.array(list_00)
array_col = numpy.array(list_col)
array_row = numpy.array(list_row)

# create dataframes
df_00 = pandas.DataFrame(data = list_00)
series_col = pandas.Series(data = list_col)
series_row = pandas.Series(data = list_row)

print(array_00)
print('Testing replacement of col 01 in array_00')
array_00[:, 1] = array_col
print(array_00)
print('type of column replacement:', type(array_col))

print('Testing replacement of row 01 in array_00')
array_00[1, :] = array_row
print(array_00)
print('type of row replacement:', type(array_row))