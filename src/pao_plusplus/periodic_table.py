import os
from contextlib import redirect_stdout, redirect_stderr
from periodic_trends import plotter
from pao_plusplus.optimize import create_optimizer
import pandas as pd
from typing import Callable
from pathlib import Path
from tqdm import tqdm
from bokeh.io import export_svg, export_png, save

def plot_periodic_table(extract_data_from_optimizer: Callable, json_directory: Path,
                        output: Path | None = None) -> None:
   
   if output is not None:
      if output.suffix not in {'.png', '.svg', '.html'}:
         raise ValueError(f"Unsupported output format: {output.suffix}")
   
   rows = []
   for json_file in tqdm(list(json_directory.glob('*.log.json'))):
      element = json_file.stem[:-4]
      optimizer = create_optimizer()
      with open(os.devnull, 'w') as fnull:
         with redirect_stdout(fnull), redirect_stderr(fnull):
            optimizer.load_state(json_file)
      rows.append([element, extract_data_from_optimizer(optimizer)])
   df = pd.DataFrame(rows, columns=["Element", "Score"])

   p = plotter(df, "Element", "Score", show = output is None, extended=False, periods_remove=[7], rescale_canvas=True)

   if output is not None:
      if output.suffix == '.png':
         export_png(p, filename=str(output))
      elif output.suffix == '.svg':
         export_svg(p, filename=str(output))
      else:
         save(p, filename=str(output))


