"""
   Author: John M. McBride
   Date: 2020.04.22

   Update: 2022.10.13

"""


import argparse
import os
from pathlib import Path
import pickle
import sys
import time

import tkinter as tk
from tkinter import Tk, Canvas, Frame, Menu
from tkinter import StringVar, IntVar, DoubleVar, BooleanVar
from tkinter import messagebox as msg
from tkinter import Button, Radiobutton, Checkbutton, Label, Entry
from tkinter import NORMAL, DISABLED, RAISED, END

try:
    from crepe import predict as crepe_predict
    CREPE_INSTALLED = True
except Exception as e:
    print(e)
    CREPE_INSTALLED = False
import librosa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import spectrogram
import simpleaudio as sa
import vamp





#----------------------------------------------------------
# Parse_arguments

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", default=False)
    parser.add_argument("--auto_re", action="store_true", default=False)
    return parser.parse_args()



#----------------------------------------------------------
# Audio tools

def wav2int(wav, c=20000):
    return (wav * c / np.max(wav)).astype(np.int16)


def clip_audio(fr, wav, beg, end, e1=0.5, e2=4., repeat=0):
    ibeg = int(beg * fr)
    iend = int(end * fr)
    wav = wav[ibeg:iend]

    envelope = np.ones(len(wav), float)
    imid = int(len(wav)/2)
    envelope[:imid] = np.linspace(0, 1, imid)**e1
    envelope[imid:] = 1 - np.linspace(0, 1, len(wav)-imid)**e2
    clipped_audio = wav * envelope

#   if repeat:
#       clipped_audio = np.concatenate([clipped_audio] * repeat, axis=0)

    return clipped_audio


### Resample sound wave at sample rate of "fr".
### Used for sonification of pitch traces
def freq_resample(F, T, fr, ignore_gap=0.05):
    out = []
    phase = 0
    for i, f in enumerate(F[:-1]):
        # If time between frequencies is greater than 'ignore_gap',
        # then set the frequency to 0
        if (T[i+1] - T[i]) > ignore_gap:
            f = 0
        start = int(T[i] * fr)
        end = int(T[i+1] * fr)
        for j in range(start, end):
           out.append(np.sin(2 * np.pi * f * (j - start) / fr + phase))
        phase += 2 * np.pi * f * (end - start) / fr
    return np.array(out)
    

#----------------------------------------------------------
# DEFINE CLASSES for F0 EXTRACTION


class f0_algorithm(object):
    def __init__(self, master):
        self.reset(master)


    def reset(self, master):
        self.fmin   = DoubleVar(master, value=50)
        self.fmax   = DoubleVar(master, value=2000)
        self.curr_params = {}
        self.time = np.array([])
        self.freq = np.array([])
        self.params = []


    def reset_params(self, params):
        self.curr_params = params
        for k, v in params.items():
            getattr(self, k).set(v)
            

class pyin(f0_algorithm):
    def __init__(self, master):
        f0_algorithm.__init__(self, master)
        self.lowamp = DoubleVar(master, 0.01)
        self.params = ['fmin', 'fmax', 'lowamp']
        self.name = 'pyin'


    def run(self, wav, fr):
        self.curr_params.update({att:getattr(self, att).get() for att in self.params})
        data = vamp.collect(wav, fr, "pyin:pyin", output='smoothedpitchtrack', parameters={'outputunvoiced': 2, 'lowampsuppression':self.lowamp.get()})
        freq = data['vector'][1]
        time = np.arange(freq.size) * float(data['vector'][0])
        idx = (freq >= self.fmin.get()) & (freq <= self.fmax.get())
        freq[idx == False] = 0

        self.time = time
        self.freq = freq


class crepe(f0_algorithm):
    def __init__(self, master):
        f0_algorithm.__init__(self, master)
        self.conf_threshold = DoubleVar(master, 0.5)
        self.params = ['fmin', 'fmax', 'conf_threshold']
        self.name = 'crepe'
        self.step_size = 5

    def run(self, wav, fr):
        self.curr_params.update({att:getattr(self, att).get() for att in self.params})

        self.time = np.nan
        self.freq = np.nan

        if CREPE_INSTALLED == False:
            return

        try:
            time, frequency, confidence, _ = crepe_predict(wav, fr, viterbi=True, step_size=self.step_size, verbose=0)
#           cuda.get_current_device().reset()
            idx = (confidence > self.conf_threshold.get()) & (frequency >= self.fmin.get()) & (frequency <= self.fmax.get())
            frequency[idx == False] = 0

            self.time = time
            self.freq = frequency
        except Exception as e:
            print(f"Exception when trying to run crepe: {e}")



### Parameter values ranges are not very intuitive..
### 
### Voicing: -2.6 <= x <= 3;
###     default: x = 0.2
###
### Minpeaksalience: 0 <= x <= 100
###     default: x = 0
class melodia(f0_algorithm):
    def __init__(self, master):
        f0_algorithm.__init__(self, master)
        self.voicing = DoubleVar(master, 0.2)
        self.minpeaksalience = DoubleVar(master, 0.0)
        self.params = ['fmin', 'fmax', 'voicing', 'minpeaksalience']
        self.name = 'melodia'


    def run(self, wav, fr):
        self.curr_params.update({att:getattr(self, att).get() for att in self.params})
        data = vamp.collect(wav, fr, "mtg-melodia:melodia", parameters={'minfqr':self.fmin.get(),
                            'maxfqr':self.fmax.get(), 'voicing':self.voicing.get(),
                            'minpeaksalience':self.minpeaksalience.get()})
        freq = data['vector'][1]
        time = np.arange(freq.size) * float(data['vector'][0])
        idx = freq > 0
        freq[idx == False] = 0

        self.time = time
        self.freq = freq


class spect(f0_algorithm):
    def __init__(self, master):
        # Not technically a f0 extraction algorithm,
        # but it's useful to inherit the 'reset' method
        f0_algorithm.__init__(self, master)
        self.window = IntVar(master, value=256)
        self.hop    = IntVar(master, value=32)
        self.params = ['window', 'hop', 'fmax']
        self.name = 'spectrogram'

    def run(self, wav, fr):
        self.curr_params.update({att:getattr(self, att).get() for att in self.params})
        f, t, spec = spectrogram(wav, fr, nperseg=self.window.get(),
                                 noverlap=self.hop.get())
        idx = f <= self.fmax.get()
        self.freq = f[idx]
        self.time = t
        self.spect = spec[idx]


#----------------------------------------------------------
# MAIN CLASS FOR GUI

class PitchTracking(Frame):

    #------------------------------------------------------
    # GUI Setup / parameter initialization methods

    def __init__(self, root):
        Frame.__init__(self, root)
        self.master = root
        self.grid_rows = 0
        self.grid_columns = 0

        # Title and size of the window
        self.master.title('Pitch Tracker')

        # Create the drop down menus
        self.create_menu()

        ### Initialise algorithm parameters
        ### and containers
        self.initialise_parameters()

        ### Create figure
        self.init_figure()

        ### Add parameter/control widgets
        self.playback_widget()
        self.annotation_widget()
        self.spect_widget()
        self.pyin_widget()
        self.crepe_widget()
        self.melodia_widget()
        self.run_save_reset_widget()


        ### Add figure to Frame
        self.graph.get_tk_widget().grid(row=0, column=3, rowspan=self.grid_rows+2, sticky='nsew')
        self.toolbar.grid(row=self.grid_rows+2, column=3)

        ### Adjust grid parameters
        self.master.grid_columnconfigure(3, weight=1)
        self.master.grid_rowconfigure(self.grid_rows+1, weight=1)


        ### Define the default file options for opening files:
        self.file_opt = {}
        self.file_opt['defaultextension'] = '.wav'
        self.file_opt['filetypes'] = [('audio files', '.wav .mp3'), ('previous analyses', '.pickle')]
        self.file_opt['initialdir'] = '.'
 
        # the window to disable when the file open/save window opens on top of it:
        self.file_opt['parent'] = root
        self.input_filename = 'None chosen'
        self.input_path = None

#       self.master.protocol("WM_DELETE_WINDOW", self.on_closing())


    def create_menu(self):
        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label='Open',command=self.onOpen)
        fileMenu.add_command(label='Quit',command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)


    def initialise_parameters(self):

        ### Intialize algorithms
        self.spect = spect(self.master)
        self.pyin = pyin(self.master)
        self.crepe = crepe(self.master)
        self.melodia = melodia(self.master)

        self.alg_list = [self.pyin, self.crepe, self.melodia]


        ### Playback Parameters
        self.play_start = StringVar(self.master, "0:00")
        self.play_end   = StringVar(self.master, "0:00")
        self.play_speed = DoubleVar(self.master, 1)
        self.audio_track = None
        self.play_curr_params = {}
        self.play_params = ['play_start', 'play_end', 'play_speed', 'play_audio',
                            'play_pitch', 'play_pitch_alg']

        self.play_text = StringVar(self.master, 'Play')
        self.play_obj = sa.PlayObject(1)
        self.play_time = None

        self.play_audio = BooleanVar(self.master, True)
        self.play_pitch = BooleanVar(self.master, False)
        self.play_on_loop = BooleanVar(self.master, False)
        self.always_set_window_time = BooleanVar(self.master, True)

        self.play_pitch_alg = IntVar(self.master, 1)


        ### Other Parameters
        self.alg_names = ['Spectrogram', 'pYIN', 'Crepe', 'Melodia']
        self.start_time = None


        ### Annotation Parameters
        self.octerr_input = IntVar(self.master, 1)
        self.octerr_str = StringVar(self.master, '')
        self.record_octerr_on = False
        self.record_text_octerr = StringVar(self.master, 'Record')
        self.cid_octerr_press = None
        self.cid_octerr_release = None

        self.delete_str = StringVar(self.master, '')
        self.record_delete_on = False
        self.record_text_delete = StringVar(self.master, 'Record')
        self.cid_delete_press = None
        self.cid_delete_release = None

        self.record_undo_on = False
        self.record_text_undo = StringVar(self.master, 'Undo')
        self.cid_undo_press = None


    def playback_widget(self):
        Label(self.master, text="Playback options"+' '*40).grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1


        Checkbutton(self.master, text='Play Audio', variable=self.play_audio).\
                    grid(row=self.grid_rows, column=0)
        Checkbutton(self.master, text='Play Pitches', variable=self.play_pitch).\
                    grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="Choose pitch trace playback algorithm").grid(row=self.grid_rows, column=1, sticky=tk.W)
        self.grid_rows += 1

        Radiobutton(self.master, text='pYIN', variable=self.play_pitch_alg, value=1).\
                    grid(row=self.grid_rows, column=0)
        Radiobutton(self.master, text='crepe', variable=self.play_pitch_alg, value=2).\
                    grid(row=self.grid_rows, column=1)
        Radiobutton(self.master, text='melodia', variable=self.play_pitch_alg, value=3).\
                    grid(row=self.grid_rows, column=2)
        self.grid_rows += 1
        Checkbutton(self.master, text='Loop', variable=self.play_on_loop).\
                    grid(row=self.grid_rows, column=1)
        Button(self.master, text=self.play_text.get(), command=self.play_wav, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2, rowspan=3)
        self.grid_rows += 3
 
        Button(self.master, text='Set time by window', command=self.set_window_time, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=1, rowspan=3)
        Checkbutton(self.master, text='Always set\ntime by window', variable=self.always_set_window_time).\
                    grid(row=self.grid_rows, column=2)
        self.grid_rows += 3

        Label(self.master, text="Start time").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.play_start).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="End time").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.play_end).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="Speed").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.play_speed).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

    def spect_widget(self):
        Label(self.master, text="f0 extraction options"+' '*40).grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1

        Label(self.master, text="Spectrogram"+' '*40).grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1

        Label(self.master, text="Window").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.spect.window).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="Hop").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.spect.hop).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="f0_max").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.spect.fmax).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1


    def pyin_widget(self):
        Label(self.master, text="pYIN").grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1

        Label(self.master, text="f0_min").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.pyin.fmin).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="f0_max").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.pyin.fmax).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="low_amp").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.pyin.lowamp).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1


    def crepe_widget(self):
        Label(self.master, text="Crepe").grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1

        Label(self.master, text="f0_min").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.crepe.fmin).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="f0_max").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.crepe.fmax).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="conf_threshold").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.crepe.conf_threshold).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1


    def melodia_widget(self):
        Label(self.master, text="Melodia").grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1

        Label(self.master, text="f0_min").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.melodia.fmin).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="f0_max").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.melodia.fmax).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="voicing").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.melodia.voicing).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="minpeaksalience").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.melodia.minpeaksalience).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1


    def annotation_widget(self):
        Label(self.master, text="Annotation controls"+' '*40).grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1

        Label(self.master, text="How many octaves?").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.octerr_input).grid(row=self.grid_rows, column=1)
        Button(self.master, textvariable=self.record_text_octerr, command=self.record_octave_error, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2)
        self.grid_rows += 1

        Label(self.master, text="Octave correcion").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.octerr_str).grid(row=self.grid_rows, column=1)
        self.grid_rows += 2

        Label(self.master, text="Delete pitches").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.delete_str).grid(row=self.grid_rows, column=1)
        Button(self.master, textvariable=self.record_text_delete, command=self.record_delete, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2)
        self.grid_rows += 3

        Button(self.master, textvariable=self.record_text_undo, command=self.record_undo, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2)
        self.grid_rows += 3


    def run_save_reset_widget(self):

        Button(self.master, text='Run Algorithms', command=self.run_analysis, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=0)
 
        Button(self.master, text='Save', command=self.save_data, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=1)
 
        Button(self.master, text='Reset', command=self.reset_parameters, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2)
 
 
    #------------------------------------------------------
    # Run analyses

    def run_analysis(self):
        # Run algorithms if parameters have changed
#       for alg in [self.spect, self.pyin, self.crepe, self.melodia]:
        for alg in [self.pyin, self.crepe, self.melodia]:
            self.run_alg_if_new_params(alg)

        # Save results checkpoint
        self.save_f0()
        self.save_params()

        # Save current window limits
        if self.initialize_graph == False:
            xlim = [a.get_xlim() for a in self.ax]
            ylim = [a.get_ylim() for a in self.ax]

        # Clear plots
        plt.clf()
        # Plot results
#       self.plot_spect(self.spect)
        for i, alg in enumerate([self.pyin, self.crepe, self.melodia]):
            self.plot_f0(alg, i+1)

        self.plot_annotation_boxes()

        # Reapply window limits
        if self.initialize_graph == False:
            [a.set_xlim(x) for a, x in zip(self.ax, xlim)]
            [a.set_ylim(y) for a, y in zip(self.ax, ylim)]
        else:
            self.initialize_graph = False


        self.graph.draw()

        print('Finished')
        

    def run_alg_if_new_params(self, alg):
        if not len(alg.curr_params):
            print(f'{alg.name}:\tRunning from scratch')
            alg.run(self.wav, self.fr)

        else:
            # If any parameter is changed from when 
            # alg.run was last called, then run alg.run
            for att, val in alg.curr_params.items():
                if getattr(alg, att).get() != val:
                    print(f'{alg.name}:\tRe-running with new parameters')
                    print(f"{att} changed from {val} to {getattr(alg, att).get()}")
                    alg.run(self.wav, self.fr)
                    return
            print(f'{alg.name}:\tNothing changed')


    #------------------------------------------------------
    # Quit / reset program

    def quit(self):
        '''
        Quit the program
        '''
        sys.exit(0)

    def on_closing(self):
        '''
        Quit the program
        '''
        if msg.askokcancel("Quit", "Do you want to quit?"):
            sys.exit(0)

    def reset_parameters(self):
        for alg in [self.spect, self.pyin, self.crepe, self.melodia]:
            alg.reset(self.master)

    #------------------------------------------------------
    # Plotting

    def init_figure(self):
        self.initialize_graph = True
        self.fig = Figure()
        self.ax = [plt.subplot2grid((4,1), (0,0), fig=self.fig)]
        for i in range(3):
            self.ax.append(plt.subplot2grid((4,1), (i+1,0), sharex=self.ax[0], fig=self.fig))
#           self.ax[i].set_xticklabels([''])

        for i in range(4):
            self.ax[i].set_title(self.alg_names[i])
            self.ax[i].set_ylabel('frequency')
            self.ax[i].plot([0,1], [0,0], '-', c='k')

        self.ax[-1].set_xlabel('time (sec)')


        self.graph = FigureCanvasTkAgg(self.fig, self.master)
        self.graph.draw()

        self.toolbar = NavigationToolbar2Tk(self.fig.canvas, self.master)
        self.toolbar.pack_forget()
        self.toolbar.update()


    def reset_figure(self, j=-1):
        if j >= 0:
            self.ax[j].clear()
            self.ax[j].set_title(self.alg_names[j])
            self.ax[j].set_ylabel('frequency')
        else:
            for i in range(4):
                self.ax[i].clear()
                self.ax[i].set_title(self.alg_names[i])
                self.ax[i].set_ylabel('frequency')
            self.ax[-1].set_xlabel('time (sec)')


    def plot_spect(self, spect):
        self.reset_figure(0)
        extent = [spect.time.min(), spect.time.max(), spect.freq.max(), spect.freq.min()]

        # Alter the spectrogram to increase contrast in the image
        smin = np.min(spect.spect[spect.spect>0])
        Z = np.log(spect.spect / smin)**4

        self.ax[0].imshow(Z, cmap='Greys', extent=extent, aspect='auto')
        self.ax[0].invert_yaxis()


    def plot_f0(self, alg, i):
        self.reset_figure(i)
        if isinstance(alg.time, (list, np.ndarray)):
            freq = self.remove_pitches(alg.time, alg.freq)
            freq = self.correct_octave_errors(alg.time, freq)
            idx = freq > 0
            self.ax[i].plot(alg.time[idx], freq[idx], 'ok', ms=3)


    def plot_annotation_boxes(self):
        self.plot_delete_annotation_boxes()
        self.plot_octerr_annotation_boxes()


    def plot_delete_annotation_boxes(self):
        ylo, yhi = self.ax[1].get_ylim()
        on, off = self.unpack_delete_str()
        for i in range(len(on)):
            # Need the min/max functions in case mouse is dragged from right to left
            xlim = [min(on[i], off[i]), max(on[i], off[i])]
            for j in range(1, 4):
                self.ax[j].fill_between(xlim, [ylo]*2, [yhi]*2, color='r', alpha=0.3)


    def plot_octerr_annotation_boxes(self):
        ylo, yhi = self.ax[1].get_ylim()
        on, off, oct_change = self.unpack_octerr_str()
        for i in range(len(on)):
            # Need the min/max functions in case mouse is dragged from right to left
            xlim = [min(on[i], off[i]), max(on[i], off[i])]
            for j in range(1, 4):
                self.ax[j].fill_between(xlim, [ylo]*2, [yhi]*2, color='g', alpha=0.3)


    #------------------------------------------------------
    # f0 correction methods

    def remove_pitches(self, time, freq):
        if not len(self.delete_str.get().strip()):
            return freq
        freq = freq.copy()
        on, off = self.unpack_delete_str()
        # Need the min/max functions in case mouse is dragged from right to left
        for i in range(len(on)):
            idx = (time >= min(on[i], off[i])) & (time <= max(on[i], off[i])) 
            freq[idx] = 0
        return freq


    def correct_octave_errors(self, time, freq):
        if not len(self.octerr_str.get().strip()):
            return freq
        freq = freq.copy()
        on, off, oct_change = self.unpack_octerr_str()
        for i in range(len(on)):
            # Need the min/max functions in case mouse is dragged from right to left
            idx = (time >= min(on[i], off[i])) & (time <= max(on[i], off[i]))
            freq[idx] = freq[idx] * 2**(oct_change[i])
        return freq


    def unpack_delete_str(self):
        if not len(self.delete_str.get().strip()):
            return [], []

        timestamps = np.array(self.delete_str.get().replace(' ','').split(','), float)
        on, off = timestamps.reshape(int(timestamps.size/2), 2).T
        return on, off


    def repack_delete_str(self, on, off):
        timestamps = np.round(np.ravel(np.array([on, off]).T), 2).astype(str)
        self.delete_str.set(', '.join(timestamps))


    def unpack_octerr_str(self):
        if not len(self.octerr_str.get().strip()):
            return [], [], []

        data = np.array(self.octerr_str.get().replace(' ','').split(','), float)
        on, off, oct_change = data.reshape(int(data.size/3), 3).T
        return on, off, oct_change


    def repack_octerr_str(self, on, off, oct_change):
        on = np.round(on, 2).astype(str)
        off = np.round(off, 2).astype(str)
        oct_change = oct_change.astype(str)
        data = np.ravel(np.array([on, off, oct_change]).T)
        self.octerr_str.set(', '.join(data))


    #------------------------------------------------------
    # f0 correction GUI methods

    def extract_xy_coords_octave_start(self, event):
        if event.inaxes is not None:
            new_str_data = [f"{event.xdata:8.2f}"]
            str_data = self.octerr_str.get().replace(' ', '').split(',') + new_str_data
            self.octerr_str.set(', '.join([s for s in str_data if s]))


    def extract_xy_coords_octave_stop(self, event):
        if event.inaxes is not None:
            new_str_data = f"{event.xdata:8.2f},{self.octerr_input.get():d}".split(',')
            str_data = self.octerr_str.get().replace(' ', '').split(',') + new_str_data
            self.octerr_str.set(', '.join([s for s in str_data if s]))


    def record_octave_error(self):
        if not self.record_octerr_on:
            self.cid_octerr_press = self.graph.mpl_connect('button_press_event', self.extract_xy_coords_octave_start)
            self.cid_octerr_release = self.graph.mpl_connect('button_release_event', self.extract_xy_coords_octave_stop)
            self.record_text_octerr.set('Stop')
            self.record_octerr_on = True
        else:
            self.graph.mpl_disconnect(self.cid_octerr_press)
            self.graph.mpl_disconnect(self.cid_octerr_release)
            self.record_text_octerr.set('Record')
            self.record_octerr_on = False

            self.plot_octerr_annotation_boxes()
            self.graph.draw()


    def extract_xy_coords_delete(self, event):
        if event.inaxes is not None:
            str_data = self.delete_str.get().replace(' ', '').split(',') + [f"{event.xdata:8.2f}"]
            self.delete_str.set(', '.join([s for s in str_data if s]))


    def record_delete(self):
        if not self.record_delete_on:
            self.cid_delete_press = self.graph.mpl_connect('button_press_event', self.extract_xy_coords_delete)
            self.cid_delete_release = self.graph.mpl_connect('button_release_event', self.extract_xy_coords_delete)
            self.record_text_delete.set('Stop')
            self.record_delete_on = True
        else:
            self.graph.mpl_disconnect(self.cid_delete_press)
            self.graph.mpl_disconnect(self.cid_delete_release)
            self.record_text_delete.set('Record')
            self.record_delete_on = False

            self.plot_delete_annotation_boxes()
            self.graph.draw()


    def extract_xy_coords_undo(self, event):
        if event.inaxes is not None:
            x = event.xdata
            self.undo_delete(x)
            self.undo_octerr(x)


    def record_undo(self):
        if not self.record_undo_on:
            self.cid_undo = self.graph.mpl_connect('button_press_event', self.extract_xy_coords_undo)
            self.record_text_undo.set('Stop')
            self.record_undo_on = True
        else:
            self.graph.mpl_disconnect(self.cid_undo)
            self.record_text_undo.set('Undo')
            self.record_undo_on = False


    def undo_octerr(self, x):
        on, off, oct_change = self.unpack_octerr_str()
        if not len(on):
            return
        idx = []
        for i in range(len(on)):
            xlo, xhi = [min(on[i], off[i]), max(on[i], off[i])]
            if not ((xlo <= x) & (x <= xhi)):
                idx.append(i)
        if len(idx):
            self.repack_octerr_str(on[idx], off[idx], oct_change[idx])
        else:
            self.octerr_str.set('')


    def undo_delete(self, x):
        on, off = self.unpack_delete_str()
        if not len(on):
            return
        idx = []
        for i in range(len(on)):
            xlo, xhi = [min(on[i], off[i]), max(on[i], off[i])]
            if not ((xlo <= x) & (x <= xhi)):
                idx.append(i)
        if len(idx):
            self.repack_delete_str(on[idx], off[idx])
        else:
            self.delete_str.set('')


    #------------------------------------------------------
    # Input / Output

    def onOpen(self, manual=''):
        """Returns a filename for your own code to open elsewhere
        """
        if self.input_path: # a text has been chosen before
            self.file_opt['initialdir'], old_input_filename = os.path.split(self.input_path)
        else:
            old_input_filename = self.input_filename
 
        OldTextFilePath = self.input_path
 
        # Define what the user will see in the open file window:
        self.file_opt['title'] = 'Choose a wav file to analyse:'
        self.file_opt['initialfile'] = ''
 
        # returns a file path and name, or '':
        if len(manual):
            self.input_path = manual
        else:
            self.input_path = tk.filedialog.askopenfilename(**self.file_opt)

        if self.input_path:  # if it's not '':
            # User didn't hit cancel:
            text_path, self.input_filename = os.path.split(self.input_path)
            ext =  os.path.splitext(self.input_path)[1].lower()
            # If starting a new file, revert to defaults
            if OldTextFilePath != None:
                self.reset_figure()
                self.graph.draw()

            if ext.lower() in ['.wav', '.mp3']:
                self.load_audiofile()
                print(f"Loaded {self.play_end.get()} minutes of audio")

        else:
            # User hit cancel. Reset path to last value:
            self.input_path = OldTextFilePath
            self.input_filename = old_input_filename

        # Save paths as Path objects
        self.path_input = Path(self.input_path)
        self.path_base = self.path_input.parent
        self.path_params = self.path_base.joinpath(f"{self.path_input.stem}_params.pkl")
        for alg in self.alg_list:
            alg.path_out = self.path_base.joinpath(f"{self.path_input.stem}_f0_{alg.name}.csv")

        # Start timer for benchmarking
        self.start_time = time.time()

        # Load previous parameters if available
        self.reload_params()
        self.reload_f0()
        


        # Set True to reset graph x/y limits
        self.initialize_graph = True

        # Run analysees
        self.run_analysis()


    def load_audiofile(self):
        ext =  os.path.splitext(self.input_path)[1].lower()
        if ext == '.wav':
            self.load_wavfile()
        elif ext == '.mp3':
            self.load_mp3file()
        self.play_end.set(self.get_recording_length())


    def load_wavfile(self, norm=True):
        fr, wav = wavfile.read(self.input_path)
        self.fr = fr
        if len(wav.shape)>1:
            self.wav = wav.mean(axis=1)
        else:
            self.wav = wav
#       if norm:
#           self.wav = self.wav.astype('float16') / self.wav.max()
        self.wav = self.wav.astype(np.float64)


    def load_mp3file(self, norm=True):
        """MP3 to numpy array"""
        a = AudioSegment.from_mp3(self.input_path)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
            y = y.mean(axis=1)
        self.wav = y
#       if norm:
#           self.wav = self.wav.astype('float16') / self.wav.max()
        self.fr = a.frame_rate
        self.wav = self.wav.astype(np.float64)


    def reload_params(self):
        if self.path_params.exists():
            params = pickle.load(open(self.path_params, 'rb'))['alg_params']
            self.pyin.reset_params(params['pyin'])
            self.crepe.reset_params(params['crepe'])
            self.melodia.reset_params(params['melodia'])


    def reload_f0(self):
        for alg in self.alg_list:
            path_out = alg.path_out
            if path_out.exists():
                time, freq = np.loadtxt(path_out, skiprows=1, delimiter=',').T
                alg.time = time
                alg.freq = freq
            # If file does not exist, reset curr_params so that
            # the algorithm will still run
            else:
                alg.curr_params = {k:np.nan for k in alg.params}


    # Save [time, f0] for each algorithm in separate csv file
    def save_f0(self):
        for alg in self.alg_list:
            path_out = alg.path_out
            # Correct f0 by removing pitches and correcting octave errors
            freq = self.remove_pitches(alg.time, alg.freq)
            freq = self.correct_octave_errors(alg.time, freq)
            print(f"Saving to {path_out}")
            np.savetxt(path_out, np.array([alg.time, freq]).T, header='time,f0', delimiter=',')


    # Save the minimum necessary parameters to recreate the results
    def save_params(self):
        alg_params = {alg.name: alg.curr_params for alg in self.alg_list}
        delete = {k:v for k, v in zip(['on', 'off'], self.unpack_delete_str())}
        octerr = {k:v for k, v in zip(['on', 'off', 'oct_change'], self.unpack_octerr_str())}
        out_dict = {'alg_params':alg_params, 'delete_pitch':delete, 'octave_error':octerr}
        print(f"Saving to {self.path_params}")
        pickle.dump(out_dict, open(self.path_params, 'wb'))


    def save_data(self):
        self.save_f0()
        self.save_params()

        delete = {k:v for k, v in zip(['on', 'off'], self.unpack_delete_str())}
        octerr = {k:v for k, v in zip(['on', 'off', 'oct_change'], self.unpack_octerr_str())}
        nd = int(len(delete['on']))
        no = int(len(octerr['on']))
        print(f"Pitches deleted in {nd} sections")
        print(f"Octave errors corrected in {no} sections")
        print(f"Time taken: {(time.time()-self.start_time):6.1f} seconds")
 

    #------------------------------------------------------
    # Playback

    def get_recording_length(self):
        length = self.wav.size / self.fr
        mins = int(length / 60)
        sec  = int(length % 60) + 1
        return f"{mins}:{sec:02d}"


    def convert_string_to_seconds(self, time_string):
        mins, sec = [int(x) for x in time_string.split(':')]
        return mins*60 + sec


    def convert_seconds_to_string(self, time_sec):
        mins = int(time_sec // 60)
        sec = int(time_sec % 60)
        return f"{mins}:{sec}"


    # Set the playback time to match the window
    def set_window_time(self):
        start, end = [max(0, x) for x in self.ax[0].get_xlim()]
        self.play_start.set(self.convert_seconds_to_string(start))
        self.play_end.set(self.convert_seconds_to_string(end))


    # Create clipped audio
    # Adjust speed if necessary using librosa
    def prepare_original_audio(self, start, end, speed):
        audio_track = clip_audio(self.fr, self.wav, start, end)
        if speed != 1.:
            audio_track = librosa.effects.time_stretch(audio_track, speed)
        return wav2int(audio_track)


    # Sonify f0 for the chosen algorithm
    def prepare_pitch_track(self, start, end, speed):
        alg = {1: self.pyin, 2: self.crepe, 3: self.melodia}[self.play_pitch_alg.get()]
        idx = (start <= alg.time) & (alg.time <= end)
        time, freq = alg.time[idx], alg.freq[idx]
        freq = self.remove_pitches(time, freq)
        freq = self.correct_octave_errors(time, freq)
        if time[0] != start:
            time = [start] + list(time)
            freq = [start] + list(freq)

        if time[-1] != end:
            time = list(time) + [end]
            freq = list(freq) + [end]

        audio_track = freq_resample(freq, time, self.fr)
        if speed != 1.:
            audio_track = librosa.effects.time_stretch(audio_track, speed)
        return wav2int(audio_track)


    def prepare_audio(self):
        start = self.convert_string_to_seconds(self.play_start.get())
        end   = self.convert_string_to_seconds(self.play_end.get())
        speed = self.play_speed.get()
        if self.play_audio.get() and not self.play_pitch.get():
            print("Preparing original audio only...")
            self.audio_track = self.prepare_original_audio(start, end, speed)

        elif not self.play_audio.get() and self.play_pitch.get():
            print("Preparing pitch trace only...")
            self.audio_track = self.prepare_pitch_track(start, end, speed)

        elif self.play_audio.get() and self.play_pitch.get():
            print("Preparing both original audio and pitch trace...")
            original = self.prepare_original_audio(start, end, speed)
            synth = self.prepare_pitch_track(start, end, speed)
            print(len(original), len(synth))
            self.audio_track = np.ascontiguousarray(np.array([original, synth]).T)


        for param in self.play_params:
            self.play_curr_params[param] = getattr(self, param).get()


    def play_wav(self):
        if self.always_set_window_time:
            self.set_window_time()

        if not self.play_obj.is_playing():

            # Prepare audio
            if isinstance(self.audio_track, type(None)):
                self.prepare_audio()
            else:
                # Prepare audio if params have changed
                for param in self.play_params:
                    if self.play_curr_params[param] != getattr(self, param).get():
                        self.prepare_audio()
                        break

            channels = len(self.audio_track.shape) 
            self.play_obj = sa.play_buffer(self.audio_track, channels, 2, self.fr)
            self.play_time = time.time()
            print("Playing audio")

        else:
            self.play_obj.stop()
            start = self.convert_string_to_seconds(self.play_start.get())
            stop_time = time.time() - self.play_time + start
            print(f"Stopped playing audio at {stop_time:6.1f} seconds")






if __name__ == '__main__':
    args = parse_args()

    
    if args.auto:
        auto_mp()
    elif args.auto_re:
        auto_reprocess(PATH_DATA2)
    else:
        root = Tk()
        app = PitchTracking(root)
        app.mainloop()

