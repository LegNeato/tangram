from .tangram_python import *

# TODO: handle missing imports?
import pandas as pd
import pyarrow as pa

def train(
  table,
  target,
  column_types=None,
  shuffle_enabled=None,
  shuffle_seed=None,
  test_fraction=None,
  comparison_fraction=None,
  autogrid=None,
	grid=None,
	comparison_metric=None
):
  if isinstance(table, pd.DataFrame):
    table = pa.Table.from_pandas(table)
  elif isinstance(table, pa.Table):
    pass
  pyarrow_arrays = []
  for column in table.itercolumns():
    pyarrow_array = (
      column._name,
      column.combine_chunks()
    )
    pyarrow_arrays.append(pyarrow_array)
  model = train_inner(
    pyarrow_arrays,
    target,
    column_types,
    shuffle_enabled,
    shuffle_seed,
    test_fraction,
    comparison_fraction,
    autogrid,
    grid,
    comparison_metric
  )
  return model