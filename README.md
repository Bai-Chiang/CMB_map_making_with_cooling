# map_making_perturbative

Here `python` is `python3`

Initialize parameters
```
python initialize_params.py
```

make sure the cache directory has enough space, the default localtion is `.cache`
``` python
cache_dir = Path('~/path/to/cache/dir').expanduser()
```

using mpi calculate multiple cases simultaneously, switch `$(nproc)` to the number of cores you want to use. Depending on `num_sample` parameter in `initialize_params.py`, using all physical cores may requires lots of RAM.
```
mpiexec -n $(nproc) python calculate.py
```
or running at background direct standard output to `.out`
```
nohup mpiexec -n $(nproc) python calculate.py > .out &
```

plot images
```
python analyze_plot.py
```

plot images in notes
```
python plot_images.py
```
