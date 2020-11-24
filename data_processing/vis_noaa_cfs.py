import numpy as np
import os

import pygrib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

data_url = "https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/time-series/2011/201104/20110401/2011040100/chi200.01.2011040100.daily.grb2"
data_name = "chi200.01.2011040100.daily.grb2"

data_dir = "./example_data/"
fig_dir = "./figs/"
data_path = data_dir + data_name

vis_fig = True
save_fig = True

def download_data():
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(data_path):
        cmd = "wget {} -O {}".format(data_url, data_path)
        os.system(cmd)

def print_info():
    grbs = pygrib.open(data_path)
    for grb in grbs:
       print(grb.level," - ",grb.name)

def plot_figure():
    plt.figure()
    grbs = grbs = pygrib.open(data_path)
    grbs = grbs.select(name='Velocity potential')
    lat,lon = grbs[0].latlons()

    if save_fig:
        os.makedirs(fig_dir, exist_ok=True)

    for ig, g in enumerate(grbs):
        data = g.values

        m = Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(), \
          urcrnrlon=lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
          resolution='c')
        x, y = m(lon,lat)
        cs = m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.jet)
        m.drawcoastlines()
        m.drawmapboundary()
        m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        plt.title('Velocity potential: {}'.format(data_name))
        
        if save_fig:
            plt.savefig(fig_dir + "{:04d}.png".format(ig))
        if vis_fig:
            plt.show()

if __name__ == '__main__':
    download_data()
    # print_info()
    plot_figure()