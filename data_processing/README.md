## Read and visualize NOAA CFS data

### Setup environments
1. Create a python 3.8 conda environment
```
conda create --name py38_727 python=3.8
```
2. Activate the environement
```
conda create --name py38_727 python=3.8
```

3. Install packages
```
pip install numpy pyproj matplotlib
```

4. Install pygrid (https://github.com/jswhit/pygrib)
```
cd ../..
git clone git@github.com:jswhit/pygrib.git
cd pygrib
cp setup.cfg.template setup.cfg
python setup.py install
cd ../HyPhyESN/data_processing/
```

5. Install mpl_toolkits.basemap

### Run demo
```
python vis_noaa_cfs.py
```