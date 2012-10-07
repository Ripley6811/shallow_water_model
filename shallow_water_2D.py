#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D shallow water model - Leap-frog method on Arakawa C-grid

Displays a running model in a GUI using TKinter.
Click "set" before "run".
Left click on the grid to add/remove obstructions.
Right click to change the location of the cross-section graphs (orange crosshair).


@SINCE: Thu Apr 05 19:43:42 2012
@VERSION: 0.5
@STATUS: Incomplete
@CHANGE: ...
@TODO:
    - Add in a bathymetry import method. Add in editing of bathymetry in the GUI.

@AUTHOR: Ripley6811
@ORGANIZATION: National Cheng Kung University, Department of Earth Sciences
@CONTACT: python@boun.cr
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Ripley6811'
__contact__ = 'python@boun.cr'
__copyright__ = ''
__license__ = ''
__date__ = 'Thu Apr 05 19:43:42 2012'
__version__ = '0.5'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

import numpy
from numpy import *  # IMPORTS ndarray(), arange(), zeros(), ones()
import matplotlib.pyplot as plt  # plt.plot(x,y)  plt.show()
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import Tkinter
from PIL import Image, ImageTk, ImageDraw
import tkFileDialog as dialog
#===============================================================================
# METHODS
#===============================================================================
# NUMPY PRINTING SETUP
set_printoptions(precision=3, suppress=True)


class shallow_water_model_GUI(Tkinter.Tk):

    def __init__(self, parent):
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent

        self.initialize()


    def initialize(self):
        #===============================================================================
        # GUI SETUP
        #===============================================================================
        # CREATE THE MENU BAR
        self.create_menu()
        # BOTTOM CONTROLS
        self.create_bottom_controls()
        # MAIN SCREEN CANVAS BINDINGS
        self.create_display_canvas()

        self.geometry( '1050x650+200+0' ) # INITIAL POSITION OF TK WINDOW

        self.update()


        # CREATE TEMPORARY BATHYMETRY. ADD IMPORT FUNCTION LATER
        self.bathy_grid = ones([100,100])*5
        self.bathy_grid[1:-1,1:-1] = -5
        self.bathy_grid[-1,1:5] = -5



        self.f = plt.Figure(figsize=(5.0,5.0), dpi=100, facecolor='w', edgecolor='w')
        self.dataPlot = FigureCanvasTkAgg(self.f, master=self.canvas)
        self.dataPlot.get_tk_widget().pack(side=Tkinter.RIGHT)



    def create_menu(self):
        menubar = Tkinter.Menu(self)
        filemenu = Tkinter.Menu(menubar)
        filemenu.add_command(label='Quit', command=self.Quit)
        menubar.add_cascade(label='File', menu=filemenu)
        self.config(menu=menubar)


    def create_bottom_controls(self):
        controls = Tkinter.Frame(self)

        Tkinter.Label(controls, text='Bathymetry file').grid(row=0, columnspan=2, sticky=Tkinter.W)#,padx=10,pady=10)
        self.bathyFilename = Tkinter.StringVar()
        self.bathyFilename.set('No file selected...')
        bathyFilenameLabel = Tkinter.Label(controls, textvariable=self.bathyFilename, width=50)
        bathyFilenameLabel.grid(row=0, column=2, columnspan=8, sticky=Tkinter.W+Tkinter.E, padx=10)


        Tkinter.Label(controls, text='Tidal Function: F(t)').grid(row=1, columnspan=2, sticky=Tkinter.W)#,padx=10,pady=10)
        self.tideFunc = Tkinter.StringVar()
        self.tideFunc.set('0.5*sin(1.*t*2.*pi/44712.)')
        tideFuncEntry = Tkinter.Entry(controls, textvariable=self.tideFunc, width=50)
        tideFuncEntry.grid(row=1, column=2, sticky=Tkinter.W+Tkinter.E, columnspan=8,padx=10)

        Tkinter.Label(controls, text='dt').grid(row=2, sticky=Tkinter.E)#,padx=10,pady=10)
        self.dt = Tkinter.StringVar()
        self.dt.set('1')
        dtEntry = Tkinter.Entry(controls, textvariable=self.dt, width=10)
        dtEntry.grid(row=2, column=1, sticky=Tkinter.W,padx=10)

        Tkinter.Label(controls, text='dx').grid(row=2, column=2, sticky=Tkinter.E)#,padx=10,pady=10)
        self.dx = Tkinter.StringVar()
        self.dx.set('50')
        dxEntry = Tkinter.Entry(controls, textvariable=self.dx, width=10)
        dxEntry.grid(row=2, column=3, sticky=Tkinter.W,padx=10)

        Tkinter.Label(controls, text='dy').grid(row=2, column=4, sticky=Tkinter.E)#,padx=10,pady=10)
        self.dy = Tkinter.StringVar()
        self.dy.set('50')
        dyEntry = Tkinter.Entry(controls, textvariable=self.dy, width=10)
        dyEntry.grid(row=2, column=5, sticky=Tkinter.W,padx=10)

        Tkinter.Label(controls, text='dH').grid(row=2, column=6, sticky=Tkinter.E)#,padx=10,pady=10)
        self.dH = Tkinter.StringVar()
        self.dH.set('0.0')
        dHEntry = Tkinter.Entry(controls, textvariable=self.dH, width=10)
        dHEntry.grid(row=2, column=7, sticky=Tkinter.W,padx=10)


        Tkinter.Label(controls, text='display interval').grid(row=3, sticky=Tkinter.E)#,padx=10,pady=10)
        self.disp_every = Tkinter.StringVar()
        self.disp_every.set('150')
        dtEntry = Tkinter.Entry(controls, textvariable=self.disp_every, width=10)
        dtEntry.grid(row=3, column=1, sticky=Tkinter.W,padx=10)

        Tkinter.Label(controls, text='k').grid(row=3, column=2, sticky=Tkinter.E)#,padx=10,pady=10)
        self.k = Tkinter.StringVar()
        self.k.set('0.022')
        dxEntry = Tkinter.Entry(controls, textvariable=self.k, width=10)
        dxEntry.grid(row=3, column=3, sticky=Tkinter.W,padx=10)

        Tkinter.Label(controls, text='NA').grid(row=3, column=4, sticky=Tkinter.E)#,padx=10,pady=10)
        self.na = Tkinter.StringVar()
        self.na.set('na')
        dyEntry = Tkinter.Entry(controls, textvariable=self.na, width=10)
        dyEntry.grid(row=3, column=5, sticky=Tkinter.W,padx=10)

        self.threeD = Tkinter.BooleanVar()
        self.threeD.set(False)
        Tkinter.Checkbutton(controls, text='3D view', variable=self.threeD, onvalue=True, offvalue=False).grid(row=3, column=6, sticky=Tkinter.E)

        self.showTide = Tkinter.BooleanVar()
        self.showTide.set(True)
        Tkinter.Checkbutton(controls, text='NA', variable=self.showTide, onvalue=True, offvalue=False).grid(row=3, column=7, sticky=Tkinter.E)

        Tkinter.Button(controls, text="SET", width=10, command=self.setup_model).grid(row=4,column=1, columnspan=2)
        Tkinter.Button(controls, text="RUN", width=10, command=self.run_sim).grid(row=4, column=3, columnspan=2)
        Tkinter.Button(controls, text="STOP", width=10, command=self.stop_sim).grid(row=4,column=5, columnspan=2)

        controls.pack(side=Tkinter.BOTTOM)


    def create_display_canvas(self):
        self.canvas = Tkinter.Canvas(self, width=850, height=800)
        self.canvas.bind("<ButtonPress-1>", self.grab_mode)
        self.canvas.bind("<B1-Motion>", self.edit_bathy)
        self.canvas.bind("<ButtonRelease-1>", self.kill_mode)
        self.canvas.bind("<ButtonRelease-3>", self.change_crossing)

        self.canvas.pack(side=Tkinter.RIGHT, expand=Tkinter.YES, fill=Tkinter.BOTH)


    def change_setup(self, event):
        print 'BANG'
        pass


    def change_crossing(self, event):
        try:
            self.x_cross = event.x * self.bathy_grid.shape[1]/500
            self.y_cross = event.y * self.bathy_grid.shape[0]/500
        except:
            pass


    def grab_mode(self, event):
        y = event.x * self.bathy_grid.shape[1]/500
        x = event.y * self.bathy_grid.shape[0]/500
        self.edit_mode = "to land" if self.bathy_grid[x,y] == -5 else "to water"

    def edit_bathy(self, event):
        y = event.x * self.bathy_grid.shape[1]/500
        x = event.y * self.bathy_grid.shape[0]/500

        if self.edit_mode == "to land":
            self.bathy_grid[x,y] = 1
        if self.edit_mode == "to water":
            self.bathy_grid[x,y] = -5

        self.display_status_array()


    def kill_mode(self, event):
        self.edit_bathy(event)
        self.edit_mode = None


    def setup_model(self):
        self.stop_sim()


        self.model = shallow_water_model(self.bathy_grid,
                                         self.dx.get(), self.dy.get(), self.dt.get(),
                                         k=float(self.k.get()),
                                         tide_function=self.tideFunc.get())
        self.x_cross = self.bathy_grid.shape[1]/2
        self.y_cross = self.bathy_grid.shape[0]/2


    def run_sim(self):
#        a = self.f.add_subplot(211)#, ylim = (-1.5,1.5))
#        b = self.f.add_subplot(212)#, ylim = (-1.5,1.5))
        try:
            self.model
        except:
            self.setup_model()

        stat_rec = [[],[],[],[]]

        self.keep_running = True
        for i, status_im, ugrid, vgrid in self.model.run_t(see_result_every=self.disp_every.get()):
            length = len(status_im)
            stat_rec[0].append(status_im[10,10])
            stat_rec[1].append(status_im[length/2,length/2])
            stat_rec[2].append(status_im[-3,3])
            stat_rec[3].append(status_im[-1,3])

            self.status_im = status_im
            # DISPLAY RESULTS
#            print status_im[1:,1:-1], i
            if isnan(status_im[10,10]):
                break

            self.display_status_array()
            self.canvas.create_text((10,10), text=i, fill='white', anchor=Tkinter.NW, tags='image')

            if self.threeD.get():
                self.plt_contours3D(status_im.copy())
            else:
                self.plt_contours(status_im.copy(), ugrid, vgrid)

            self.dataPlot.show()

            self.update()
            if not self.keep_running:
                break
        plt.figure()
        print stat_rec
        labels = ['(3,-3)','Middle','(-3,3)','(-1,3)']
        for i in xrange(len(stat_rec)):
            plt.plot(stat_rec[i], label=labels[i])
        plt.legend()
        plt.show()


    def display_status_array(self):
#        self.display_image = array2GREY(status_im)
        self.display_image = array2COLOR(self.status_im, self.x_cross, self.y_cross, self.bathy_grid)
        self.canvas.delete('image')
        self.canvas.create_image((0,0), image=self.display_image , anchor=Tkinter.NW, tags='image' )


    def stop_sim(self):
        self.keep_running = False


    def plt_contours(self, z_data, ugrid, vgrid):
        self.f.clear()
        axX = self.f.add_axes([0.0,0.75,0.7,0.2])#, ylim=(-1.,1))
        axY = self.f.add_axes([0.75,0.0,0.2,0.7])#, xlim=(-1.,1))
        axX.grid(True)
        axY.grid(True)
        CS = self.f.add_axes([0.0,0.0,0.7,0.7])
        CS.imshow(z_data[1:-1,1:-1], interpolation='bilinear')#, origin='lower'),
#                cmap=cm.copper)
        CS = CS.contour( z_data[1:-1,1:-1], colors='k')
        x = arange(len(z_data[self.y_cross]))
        axX.fill_between(x[1:-1], average(z_data[1:-1,1:-1],0), color='lightblue')
        axX.plot(x[1:-1], z_data[self.y_cross,1:-1], 'k')
        y = arange(len(z_data[self.x_cross]))
        axY.fill_betweenx( y[1:-1], average(z_data[1:-1,1:-1],1)[::-1], color='lightblue')
        axY.plot(z_data[1:-1, self.x_cross][::-1], y[1:-1],  'k')

        plt.clabel(CS, inline=1, fontsize=10)

        #######################
#        plt.title('Simplest default with labels')

#        plt.figure()
#        plt.quiver(-ugrid[::-1], vgrid[::-1])
#        plt.show()


    def plt_contours3D(self, z_data):
        self.f.clear()
        axX = self.f.add_axes([0.0,0.75,0.7,0.2])#, ylim=(-1.,1))
        axY = self.f.add_axes([0.75,0.0,0.2,0.7])#, xlim=(-1.,1))
        axX.grid(True)
        axY.grid(True)
        x = arange(len(z_data[self.y_cross]))
        y = arange(len(z_data[self.x_cross]))
        X, Y = meshgrid(x[1:-1], y[1:-1])
        CS = self.f.add_axes([0.0,0.0,0.7,0.7], projection='3d')
        CS.plot_wireframe(X,Y,z_data[1:-1,1:-1][::-1], rstride=2, cstride=2)
        axX.fill_between(x[1:-1], average(z_data[1:-1,1:-1],0), color='lightblue')
        axX.plot(x[1:-1], z_data[self.y_cross,1:-1], 'k')
        axY.fill_betweenx( y[1:-1], average(z_data[1:-1,1:-1],1)[::-1], color='lightblue')
        axY.plot(z_data[1:-1, self.x_cross][::-1], y[1:-1],  'k')
#        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Simplest default with labels')


    def Quit(self):
        self.destroy()
        self.quit()


def array2GREY(orig_array):
    new_image = orig_array.copy()
    minval = numpy.min(new_image[1:-1,1:-1])
    new_image -= minval
    maxval = numpy.max(new_image[1:-1,1:-1])
    new_image *= 255/maxval
    new_image = Image.fromarray(new_image).resize((500,500))
    new_image = new_image.convert('RGB')
    print 'pix', new_image.getpixel((3,3))

    return ImageTk.PhotoImage(new_image )


def array2COLOR(orig_array, x_cross, y_cross, bathy_grid):
    new_image = orig_array.copy()
    new_image -= new_image[1:-1,1:-1].mean()
    maxval = numpy.max(numpy.absolute(new_image[1:-1,1:-1]))
    r_band = where(new_image > 0, new_image*255/maxval, 0)
    r_band[y_cross,:] = 255
    r_band[:,x_cross] = 255
    r_band[where(bathy_grid >= 0)] = 149
    g_band = where(new_image < 0, -new_image*255/maxval, 0)
    g_band[y_cross,:] = 127
    g_band[:,x_cross] = 127
    g_band[where(bathy_grid >= 0)] = 69
    b_band = where(numpy.absolute(new_image) < maxval/2, 100-numpy.absolute(new_image)*255/maxval, 0)
    b_band[y_cross,:] = 0
    b_band[:,x_cross] = 0
    b_band[where(bathy_grid >= 0)] = 53
    r_band = Image.fromarray(r_band).convert('L')
    g_band = Image.fromarray(g_band).convert('L')
    b_band = Image.fromarray(b_band).convert('L')
    new_image = Image.merge('RGB', [r_band, g_band, b_band]).resize((500,500))
    return ImageTk.PhotoImage(new_image)


class shallow_water_model:

    def __init__(self, bathymetry_grid, dx, dy, dt, dH=0.0, tide_function=None, k=0.022, g=-9.80665):
        self.H = bathymetry_grid - float(dH)
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.k = k
        self.g = g

        self.u = zeros(self.H.shape)
        self.v = zeros(self.H.shape)
        self.z = zeros(self.H.shape)

        self.q_time = 0
        self.z_time = 0

        self.tide_pts = self.get_tide_control_pts( self.H )

        if tide_function:
            self.eval_tide(tide_function)
        else:
            self.eval_tide('1*sin(1.*t*2.*pi/44712.)')


    def get_tide_control_pts(self, bathy_grid):
        '''Get a list of indexes along map edges where tide will be controlled.'''
        col = where(bathy_grid[0] < 0)[0]
        row = zeros(len(col), int)

        col = append(col, where(bathy_grid[-1] < 0)[0])
        row = append(row, [-1]*(len(col)-len(row)))

        row = append(row, where(bathy_grid[:,0] < 0)[0])
        col = append(col, [0]*(len(row)-len(col)))

        row = append(row, where(bathy_grid[:,-1] < 0)[0])
        col = append(col, [-1]*(len(row)-len(col)))

        return row.astype(int), col.astype(int)


    def eval_tide(self, arg):
        if isinstance(arg, str):
            self.tide_function = arg
        elif isinstance(arg, int):
            t = arg
            try:
                return eval(self.tide_function)
            except TypeError:
                print "Error: Tidal function not set or did not use 't' as time variable."
        else:
            raise TypeError, 'Error: Must pass a new function string or a time to evaluate.'


    def tide_pos(self, new_pos=None):
        '''Set and get tidal position.'''
        if not new_pos:
            return self.H[self.tide_pts]
        for each in self.tide_pts:
            self.H[each] = new_pos


    def run_t(self, end_time=None, see_result_every=1):
        '''Use 'yield' to pause every 't' iterations?'''
        see_result_every = int(see_result_every)
        t = 0
        g = self.g
        tx = 2. * float(self.dt)/float(self.dx)
        ty = 2. * float(self.dt)/float(self.dy)
        u = self.u
        v = self.v
        z = self.z
        H = self.H
        # LOOP OVER TIME
        while True:
            # INCREMENT TO NEW EVALUATION TIME
            t += 1

            # UPDATE THE CONTROLLED TIDE
            self.z[self.tide_pts] = self.eval_tide(t)

            # ALTERNATE BETWEEN u AND z UPDATES
            if t % 2: # if t is odd
                utmp = u.copy()
                vtmp = v.copy()
                # CALCULATE ALL INTERIOR u'S
                u[1:-1,:-1] -= g * tx * (z[1:-1,1:] - z[1:-1,:-1])
                u[1:-1,:-1] += 2*g* float(self.dt) * u[1:-1,:-1] * sqrt(utmp*utmp + vtmp*vtmp)[1:-1,:-1] * self.k
                # MAKE CORRECTIONS TO BORDER u'S
                u[1:-1,:-1] *= where(H[1:-1,:-1] <= 0, True, False) * where(H[1:-1,1:] <= 0, True, False)
                # CALCULATE ALL INTERIOR v'S
                v[:-1,1:-1] -= g * ty * (z[1:,1:-1] - z[:-1,1:-1])
                v[:-1,1:-1] += 2*g* float(self.dt) * v[:-1,1:-1] * sqrt(utmp*utmp + vtmp*vtmp)[:-1,1:-1] * self.k
                # MAKE CORRECTIONS TO BORDER v'S
                v[:-1,1:-1] *= where(H[:-1,1:-1] <= 0, True, False) * where(H[1:,1:-1] <= 0, True, False)

            else:
                ztmp = z.copy()
                # CALCULATE u CONTRIBUTION TO z
                z[1:-1,1:-1] -= tx * u[1:-1,1:-1] * (H[1:-1,1:-1]+0.5*(ztmp[1:-1,2:] + ztmp[1:-1,1:-1]))
                z[1:-1,1:-1] += tx * u[1:-1,:-2] * (H[1:-1,:-2]+0.5*(ztmp[1:-1,1:-1] + ztmp[1:-1,:-2]))
                # CALCULATE v CONTRIBUTION TO z
                z[1:-1,1:-1] -= ty * v[1:-1,1:-1] * (H[1:-1,1:-1]+0.5*(ztmp[2:,1:-1] + ztmp[1:-1,1:-1]))
                z[1:-1,1:-1] += ty * v[:-2,1:-1] * (H[:-2,1:-1]+0.5*(ztmp[1:-1,1:-1] + ztmp[:-2,1:-1]))



            # YIELD OUTPUT AT REQUESTED INTERVAL
            if t % see_result_every == 0:
                yield t, self.z, self.u, self.v

            # END CONDITION TEST
            if end_time:
                if t >= end_time:
                    break


if __name__ == '__main__':
    app = shallow_water_model_GUI(None)
    app.title('Shallow Water Model - Leap-frog C-grid')
    app.mainloop()