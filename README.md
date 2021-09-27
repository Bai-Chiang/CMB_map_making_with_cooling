# Cooling Improves Cosmic Microwave Background Map-Making When Low-Frequency Noise is Large

You could find notes in `notes` directory.

The codes are in the directory `codes`

## run the code
- Change directory to `codes`
  ```
  cd codes
  ```
- read and change the parameters in `initialize_params.py`
- Initialize parameters
  ```
  python initialize_params.py
  ```
  
  By default the program store computed results in the `cache` directory.
  To create a symbolic link to the storage drive
  ```
  ln -s /path/to/storage/dir cache
  ```

- simulation and calculation
  using mpi calculate multiple cases simultaneously, change `$(nproc)` to the number of cores you want to use.
  Depending on `num_sample` parameter in `initialize_params.py`, using all physical cores may requires lots of RAM.
  And don't spawn more threads than total cases.
  ```
  mpiexec -n $(nproc) python calculate.py
  ```
  or running at background direct standard output to `out`
  ```
  nohup mpiexec -n $(nproc) python calculate.py > out 2>&1 &
  ```
  or single thread
  ```
  python calculate.py
  ```
  or with 6 threads running at background
  ```
  nohup mpiexec -n 6 python calculate.py > out 2>&1 &
  ```

- plot results
  plot images for analysis
  ```
  python analyze_plot.py
  ```
  
  plot images in paper
  ```
  python plot_paper_images.py
  ```
