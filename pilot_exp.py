# !/usr/bin/env python3.7
# -*- coding: utf-8 -*-
""" 
Created on 09.03.22

Experiment script for color reference repulsion experiments.

The main experiment should set task_type as 3 (a mix of judgment only and dual tasks).

Run from bash:
python3 exp/pilot_exp.py \
-s s1 \
-t 3 \
-c config/configs.yaml \
-p config/params_pair1.yaml \
-b 1 \
-r test_data

Version 1: only ends of colorbar presented during the stimulus presentation

@author yannansu
"""

import numpy as np
from bisect import bisect_left
from psychopy import monitors, visual, core, event
import time
import argparse
from pyiris.colorspace import ColorSpace
from yml2dict import yml2dict
from write_data import WriteData, WriteActivity


class Exp:
    def __init__(self, subject, task_type, cfg_file, par_file, block_num, res_dir, feedback=False):
        """

        :param subject:     subject name
        :param task_type:   0: estimation only, without ref; 1: estimation only, with ref; 2: judgment only; 3: n% judgment + (1-n)% dual task (judgment + estimation)
        :param cfg_file:    config file name
        :param par_file:    parameter file name
        :param block_num:   number of blocks
        :param res_dir:     result directory
        :param feedback:    True in practice sessions
        """
        self.subject = subject
        self.task_type = task_type
        self.cfg_file = cfg_file
        self.par_file = par_file
        self.cfg = yml2dict(cfg_file)
        self.par = yml2dict(par_file)
        self.block_num = block_num
        self.res_dir = res_dir
        self.feedback = feedback

        # Monitor settings
        mon_settings = yml2dict(self.cfg['monitor_settings_file'])
        monitor = monitors.Monitor(name=mon_settings['name'], distance=mon_settings['preferred_mode']['distance'])

        # Only need to run and save the monitor setting once!!!
        # The saved setting json is stored in ~/.psychopy3/monitors and should have a backup in config/resources/
        monitor.setSizePix((mon_settings['preferred_mode']['width'],
                            mon_settings['preferred_mode']['height']))
        monitor.setWidth(mon_settings['size']['width'] / 10.)
        monitor.saveMon()

        # Specify the calibration date 20220414 corresponding to the isoluminance data
        subject_path = self.cfg['subject_isolum_directory'] + 'colorspace_' + subject + '_10bit_20220414.json'
        self.bit_depth = self.cfg['depthBits']

        self.colorspace = ColorSpace(bit_depth=self.bit_depth,
                                     chromaticity=self.cfg['chromaticity'],
                                     calibration_path=self.cfg['calibration_file'],
                                     subject_path=subject_path)

        # self.gray = self.colorspace.lms_center  # or use lms center
        self.gray = np.array([self.colorspace.gray_level, self.colorspace.gray_level, self.colorspace.gray_level])
        self.colorspace.create_color_list(hue_res=0.05,
                                          gray_level=self.colorspace.gray_level)  # Make sure the reslution is fine enough - 0.05 should be good
        self.colorlist = self.colorspace.color_list[0.05]
        self.gray_pp = self.colorspace.color2pp(self.gray)[0]
        self.win = visual.window.Window(size=[mon_settings['preferred_mode']['width'],
                                              mon_settings['preferred_mode']['height']],
                                        monitor=monitor,
                                        units=self.cfg['window_units'],
                                        fullscr=True,
                                        colorSpace='rgb',
                                        color=self.gray_pp,
                                        mouseVisible=False)

    def run_exp(self):

        texts = self.make_texts()
        if self.task_type in [0, 1]:
            texts["type1"].draw()
            self.win.flip()
        if self.task_type == 2:
            texts["type2"].draw()
            self.win.flip()
        if self.task_type == 3:
            texts["type3"].draw()
            self.win.flip()
        event.waitKeys()

        # Initial activity log file: create one if it does not exist
        InitActivity = WriteActivity(self.subject, self.res_dir)
        # Iterate over blocks within session
        for block_i in np.arange(self.block_num):
            # Initialize data files for this block
            idx = time.strftime("%Y%m%dT%H%M", time.localtime())  # index as current date and time
            InitData = WriteData(self.subject, idx, self.res_dir)
            data_file = InitData.head(self.cfg_file, self.par_file)

            # Create a 2d array (n_ref x n_relative_stim, 2)
            # In each pair: (ref, relative_stim)
            ref_stim_pairs = np.column_stack([
                np.repeat(self.par['ref_hue'], len(self.par['stimulus_relative_hue'])),
                np.tile(self.par['stimulus_relative_hue'], len(self.par['ref_hue']))])
            ref_stim_pairs_repeats = np.repeat(ref_stim_pairs, self.par['n_repeats'], axis=0)

            only_e = np.array([0, 1])
            only_j = np.array([1, 0])
            j_and_e = np.array([1, 1])
            if self.task_type in [0, 1]:
                trial_types = np.tile(only_e, (len(ref_stim_pairs_repeats), 1))
            if self.task_type == 2:
                trial_types = np.tile(only_j, (len(ref_stim_pairs_repeats), 1))
            if self.task_type == 3:
                n_11_trials = int(np.round(self.par['n_repeats'] / self.par['n_split']))
                n_10_trials = int(self.par['n_repeats'] - n_11_trials)

                trial_patterns = np.concatenate([np.tile(j_and_e, (n_11_trials, 1)),
                                                 np.tile(only_j, (n_10_trials, 1))])
                trial_types = np.tile(trial_patterns, (len(ref_stim_pairs), 1))

            # 4 columns in each pair: ref, relative_stim, if_judge, if_estimate
            all_pairs = np.column_stack((ref_stim_pairs_repeats, trial_types)).astype(list)
            np.random.shuffle(all_pairs)

            trialCount = 0
            ro_num = len(all_pairs)
            # block_bias = []

            so_i = 0
            while so_i < ro_num:
                discard_this_trial = False
                trialCount += 1
                texts['trial_count'].text = f"trial {trialCount} of {ro_num} trials"
                this_pair = all_pairs[so_i]
                ref_hue = this_pair[0]
                stim_hue = ref_hue + this_pair[1]

                dat = {'trialCount': trialCount,
                       'task_type': self.task_type,
                       'stimulus': stim_hue,
                       'ref': this_pair[0],
                       'relative_stimulus': this_pair[1],
                       'right_to_ref': None,
                       'RT_ref': None,
                       'ref_judge_correct': None,
                       'response': None,
                       'bias': None,
                       'RT_estimate': None
                       }

                """
                # Option 1: plot color bar and ref patch separately
                # Make reference
                ref = self.make_ref_patch(ref_hue)
                # Make colorbars
                colorbar_hues = np.linspace(ref_hue - self.cfg['colorbar']['half_range'],
                                            ref_hue + self.cfg['colorbar']['half_range'], self.cfg['colorbar']['num'])
                closest_theta, closest_rgb = zip(*[self.closest_hue(v) for v in colorbar_hues])
                xlim = self.cfg['colorbar']['size'][0]/2.
                colorbars = self.make_color_bar(rgbs=np.array(closest_rgb), xlims=[-xlim, xlim])
                """

                """
                # Option 2: plot ref patch between two color bars
                # Make reference
                ref = self.make_ref_patch(ref_hue)
                # Make colorbars
                colorbar_hues = np.linspace(ref_hue - self.cfg['colorbar']['half_range'],
                                            ref_hue + self.cfg['colorbar']['half_range'], self.cfg['colorbar']['num'])
                closest_theta, closest_rgb = zip(*[self.closest_hue(v) for v in colorbar_hues])
                xlims_bar1 = [
                    self.cfg['ref']['pos'][0] - self.cfg['ref']['size'][0] / 2. - self.cfg['colorbar']['size'][0],
                    self.cfg['ref']['pos'][0] - self.cfg['ref']['size'][0] / 2.]
                xlims_bar2 = [self.cfg['ref']['pos'][0] + self.cfg['ref']['size'][0] / 2.,
                              self.cfg['ref']['pos'][0] + self.cfg['ref']['size'][0] / 2. +
                              self.cfg['colorbar']['size'][
                                  0]]
                colorbar1 = self.make_color_bar(rgbs=np.split(np.array(closest_rgb), 2)[0], xlims=xlims_bar1)
                colorbar2 = self.make_color_bar(rgbs=np.split(np.array(closest_rgb), 2)[1], xlims=xlims_bar2)
                colorbars = colorbar1 + colorbar2
                """

                # Option 3: plot reference as the central bin in the colorbar
                colorbar_hues = np.linspace(ref_hue - self.cfg['colorbar']['half_range'],
                                            ref_hue + self.cfg['colorbar']['half_range'], self.cfg['colorbar']['num']+1)
                closest_theta, closest_rgb = zip(*[self.closest_hue(v) for v in colorbar_hues])
                xlim = self.cfg['colorbar']['size'][0]/2.
                if self.task_type == 0:
                    show_ref = False
                else:
                    show_ref = True
                colorbars = self.make_color_bar(rgbs=np.array(closest_rgb), xlims=[-xlim, xlim], make_ref=show_ref)
                colorends = [colorbars[0], colorbars[self.cfg['colorbar']['half_num']], colorbars[-1]]  # only include two ends and the reference bin

                # Make stimulus
                stim = self.make_stim_array(stim_hue, self.par['noise'])

                # Make mask
                mask = self.make_checkerboard_mask()

                # Make cues
                judge_cue = self.make_judge_cue()
                estimate_cue = self.make_estimate_cue()

                # mouse = event.Mouse(win=self.win, visible=False)

                # Trial starts
                # Present reference only
                judge_cue.draw()
                [c.draw() for c in colorbars]
                # ref.draw()
                texts['trial_count'].draw()
                self.win.flip()
                core.wait(self.cfg['ref']['duration'])

                # Present reference + stimulus
                # [c.draw() for c in colorbars]
                [c.draw() for c in colorends]
                # ref.draw()
                stim.draw()
                texts['trial_count'].draw()
                self.win.flip()
                core.wait(self.cfg['stim']['duration'])

                # Present mask
                mask.draw()
                self.win.flip()
                core.wait(self.cfg['mask']['duration'])

                # if judgment is required
                if this_pair[2] == 1:
                    # Present reference + judge hue
                    [c.draw() for c in colorbars]
                    # ref.draw()
                    judge_cue.draw()
                    texts['trial_count'].draw()
                    self.win.flip()

                    reactClock = core.Clock()

                    right_to_ref = None
                    while right_to_ref is None:
                        for wait_keys in event.waitKeys(keyList=['left', 'right', 'escape']):
                            key_press = wait_keys
                            if key_press == 'left':
                                right_to_ref = 0  # judge as more left
                            elif key_press == 'right':
                                right_to_ref = 1  # judge as more right
                            elif key_press == 'escape':
                                texts["confirm_escape"].draw()
                                self.win.flip()
                                for key_press in event.waitKeys():
                                    if key_press == 'y':
                                        # Save escape info
                                        InitActivity.write(idx, block_i, self.cfg_file, self.par_file, data_file, status='escape')
                                        core.quit()
                                    #TODO: add 'N' to escape confrimation

                    RT_ref = np.round(reactClock.getTime(), 2)
                    if RT_ref > self.cfg['ref']['RT_limit']:
                        discard_this_trial = True

                    # Save escape info
                    if event.getKeys('escape'):
                        texts["confirm_escape"].draw()
                        self.win.flip()
                        if event.getKeys('y'):
                            InitActivity.write(idx, block_i, self.cfg_file, self.par_file, data_file, status='escape')
                            core.quit()
                        else:
                            continue

                    dat['right_to_ref'] = right_to_ref
                    dat['RT_ref'] = float(RT_ref)

                    dis_to_ref = this_pair[1]
                    if (dis_to_ref > 0 and right_to_ref == 1) or (dis_to_ref < 0 and right_to_ref == 0) or (
                            dis_to_ref == 0):
                        dat['ref_judge_correct'] = 1
                    else:
                        dat['ref_judge_correct'] = 0

                    if self.feedback or self.task_type == 3:
                        if dat['ref_judge_correct'] == 1:
                            texts["practice_feedback"].text = f"correct"
                            texts["practice_feedback"].color = 'black'
                        if dat['ref_judge_correct'] == 0:
                            texts["practice_feedback"].text = f"incorrect"
                            texts["practice_feedback"].color = 'red'

                # Estimate by picking a color from the colorbar
                if this_pair[3] == 1:
                    # Present reference only
                    [c.draw() for c in colorbars]
                    # ref.draw()
                    estimate_cue.draw()
                    texts['trial_count'].draw()
                    self.win.flip()

                    mouse = event.Mouse(win=self.win, visible=True, newPos=(0, 0))
                    estimated = False
                    reactClock = core.Clock()
                    mouse.visible = True
                    while not estimated:
                        for cidx, element in enumerate(colorbars):
                            if mouse.isPressedIn(element):
                                estimated = True
                                response = closest_theta[cidx]
                                rt_estimate = reactClock.getTime()
                                bias = response - stim_hue
                                if bias < -180:
                                    bias += 360
                                elif bias > 180:
                                    bias = 360 - bias

                                # Record responses and save
                                dat['response'] = float(np.round(response, 2))
                                dat['bias'] = float(np.round(bias, 2))
                                dat['RT_estimate'] = float(np.round(rt_estimate, 2))

                                if self.task_type in [0, 1] and self.feedback:
                                    texts["practice_feedback"].text = f"Error: {bias}"

                    # Show warning if the bias is too large
                    if discard_this_trial and 'bias' in dat.keys():
                        if dat['RT_estimate'] > self.cfg['stim']['RT_limit']:
                            texts["warning_rt"].draw()
                        else:
                            texts["warning_error"].draw()
                        self.win.flip()
                        core.wait(self.cfg['ITI'])

                dat['discard_this_trial'] = discard_this_trial
                InitData.write(trialCount, 'trial', dat)

                # Save escape info
                if event.getKeys('escape'):
                    texts["confirm_escape"].draw()
                    self.win.flip()
                    for key_press in event.waitKeys():
                        if key_press == 'y':
                            InitActivity.write(idx, block_i, self.cfg_file, self.par_file, data_file,
                                               status='escape')
                            core.quit()
                        else:
                            continue

                # Show feedback if it is a practice block
                if self.feedback is True or (self.task_type == 3 and this_pair[3] == 0):
                    texts["practice_feedback"].draw()
                    self.win.flip()
                    core.wait(self.cfg['ITI'])

                mouse = event.Mouse(win=self.win, visible=False)
                mask.draw()
                core.wait(self.cfg['ITI'])
                self.win.flip()

                # Whether discard this trial and repeat the stimulus
                if discard_this_trial:
                    so_i -= 1
                so_i += 1

                # Save block info
            if self.feedback:
                InitActivity.write(idx, block_i, self.cfg_file, self.par_file, data_file, status='practice')
            else:
                InitActivity.write(idx, block_i, self.cfg_file, self.par_file, data_file, status='completed')

                # Present message after each block
            pause = visual.TextStim(self.win,
                                    f"{block_i + 1} of {self.block_num} blocks are completed. \n "
                                    f"Press any key to continue the next block :)",
                                    color=self.cfg['text']['color'],
                                    pos=(0, 0),
                                    height=self.cfg['text']['height'])
            pause.draw()
            self.win.flip()
            event.waitKeys()

        # End message when the session is completed
        texts["goodbye"].draw()
        self.win.flip()
        event.waitKeys()

    def closest_hue(self, theta):
        """
        Tool function:
        Given a desired hue angle, to find the closest hue angle and the corresponding rgb value.

        :param theta:   desired hue angle (in degree)
        :return:        closest hue angle, closest rgb values
        """

        hue_angles = np.array(self.colorlist['hue_angles'])
        if theta < 0:
            theta += 360
        if theta >= 360:
            theta -= 360
        closest_theta, pos = np.array(self.take_closest(hue_angles, theta))
        closest_rgb = self.colorlist['rgb'][pos.astype(int)]
        closest_rgb = self.colorspace.color2pp(closest_rgb)[0]
        return np.round(closest_theta, 2), closest_rgb

    def take_closest(self, arr, val):
        """
        Tool function:
        Assumes arr is sorted. Returns closest value to val (could be itself).
        If two numbers are equally close, return the smallest number.

        :param arr:   sorted array
        :param val:   desired value
        :return:      [closest_val, closest_idx]
        """
        pos = bisect_left(arr, val)
        if pos == 0:
            return [arr[0], pos]
        if pos == len(arr):
            return [arr[-1], pos - 1]
        before = arr[pos - 1]
        after = arr[pos]
        if after - val < val - before:
            return [after, pos]
        else:
            return [before, pos - 1]

    def make_ref_patch(self, theta):
        """
        Create a color patch as reference.

        :param theta:       hue angle
        :return:            a colorpatch as a psychopy Rect stimulus
        """

        closest_theta, closest_rgb = self.closest_hue(theta)
        color_patch = visual.Rect(win=self.win,
                                  width=self.cfg['ref']['size'][0],
                                  height=self.cfg['ref']['size'][1],
                                  pos=self.cfg['ref']['pos'],
                                  lineWidth=0,
                                  fillColor=closest_rgb)
        return color_patch

    def make_color_bar(self, rgbs, xlims, make_ref=False):
        """
        Create a color bar stimulus covering a given hue angle range.

        :param rgbs:        rgb values of colors in the bar
        :param xlims:       x position lims of the bar
        :param make_ref:    whether include reference inside the bar
        :return:            a colorbar as a psychopy ImageStim stimulus
        """
        num = len(rgbs)
        element_width = self.cfg['colorbar']['size'][0] / num
        # ypos = self.cfg['ref']['pos'][1]
        ypos = self.cfg['colorbar']['ypos']
        pos = [[x, ypos] for x
               in np.linspace(xlims[0] + element_width / 2., xlims[1] - element_width / 2., num, endpoint=True)]

        colorbar = []
        for idx in np.arange(num):
            element = visual.Rect(win=self.win,
                                  width=self.cfg['colorbar']['size'][0] / num,
                                  height=self.cfg['colorbar']['size'][1],
                                  lineWidth=0,
                                  fillColor=rgbs[idx])
            element.pos = pos[idx]

            # include reference in the colorbar to make it click-able
            if make_ref is True:
                if idx == self.cfg['colorbar']['half_num']:
                    element.height = self.cfg['ref']['size'][1]

            colorbar.append(element)

        return colorbar

    def make_stim_array(self, theta, width):
        """
        Create a stimulus array of color circles.

        :param theta:   the mean hue angle of all hues in the stimulus array
        :param width:   the half width of a uniform distribution in the stimulus array
        :return:        an array of circular patches as a Psychopy ElementArrayStim stimulus
        """
        n = int(np.sqrt(self.cfg['stim']['element_nmb']))
        pos = [[x, y]
               for x in np.linspace(self.cfg['stim']['array_xlim'][0], self.cfg['stim']['array_xlim'][1], n)
               for y in np.linspace(self.cfg['stim']['array_ylim'][0], self.cfg['stim']['array_ylim'][1], n)]
        patch = visual.ElementArrayStim(win=self.win,
                                        fieldSize=self.cfg['stim']['field_size'],
                                        xys=pos,
                                        nElements=self.cfg['stim']['element_nmb'],
                                        elementMask='circle',
                                        elementTex='None',
                                        sizes=self.cfg['stim']['element_size'])

        # Fill the circle array with colors:
        # Sample from uniform distribution with equivalent spaces
        noise = np.linspace(theta - width, theta + width, int(self.cfg['stim']['element_nmb']), endpoint=True)
        # Shuffle the position of the array
        np.random.shuffle(noise)
        angle, rgbs = zip(*[self.closest_hue(theta=n) for n in noise])
        patch.colors = rgbs

        return patch

    def make_checkerboard_mask(self):
        horiz_n = 35
        vertic_n = 25
        rect = visual.ElementArrayStim(self.win,
                                       units='norm',
                                       nElements=horiz_n * vertic_n,
                                       elementMask=None,
                                       elementTex=None,
                                       sizes=(2 / horiz_n, 2 / vertic_n))
        rect.xys = [(x, y)
                    for x in np.linspace(-1, 1, horiz_n, endpoint=False) + 1 / horiz_n
                    for y in np.linspace(-1, 1, vertic_n, endpoint=False) + 1 / vertic_n]

        rect.colors = [self.closest_hue(theta=x)[1]
                       for x in
                       np.random.randint(0, high=360, size=horiz_n * vertic_n)]

        return rect

    def make_judge_cue(self):
        # Define black fixation cue for boundary judge
        judge_cue = visual.Circle(self.win,
                                  pos=self.cfg['cue']['pos'],
                                  radius=self.cfg['cue']['radius'],
                                  lineWidth=self.cfg['cue']['lineWidth'],
                                  fillColor=self.cfg['cue']['judge_fillColor'])
        return judge_cue

    def make_estimate_cue(self):
        # Define black fixation cue for boundary judge
        estimate_cue = visual.Circle(self.win,
                                     pos=self.cfg['cue']['pos'],
                                     radius=self.cfg['cue']['radius'],
                                     lineWidth=self.cfg['cue']['lineWidth'],
                                     fillColor=self.cfg['cue']['estimate_fillColor'])
        return estimate_cue

    def make_texts(self):
        texts = {}
        # Define welcome message
        texts["type1"] = visual.TextStim(self.win,
                                         text=f"Welcome! \n \n"
                                              f"Please estimate the average color of the stimulus array and select it from the color bar. \n"
                                              f"Use mouse click to select. \n \n"
                                              f"Ready? \n"
                                              f"Press any key to start this session :)",
                                         pos=(0, 0),
                                         color=self.cfg['text']['color'],
                                         height=self.cfg['text']['height'])

        texts["type2"] = visual.TextStim(self.win,
                                         text=f"Welcome! \n \n"
                                              f"Please judge whether the average color is closer to the color at the left or right end relative to the central reference color, \
                                              according to the color sequence. Press left or right key to judge. \n"
                                              f"Ready? \n"
                                              f"Press any key to start this session :)",
                                         pos=(0, 0),
                                         color=self.cfg['text']['color'],
                                         height=self.cfg['text']['height'])

        texts["type3"] = visual.TextStim(self.win,
                                         text=f"Welcome! \n \n"
                                              f"Please estimate the average color of the stimulus array. \n"
                                              f"When a black fixation point is presented, \
                                              judge whether the average color is closer to the color at the left or right end relative to the central reference color, according to the color sequence. \
                                              Press left or right key to judge. \n"
                                              f"Then you will be presented with a judgement feedback (trial completed) or a white fixation point (estimation required). \n"
                                              f"If a white fixation point is presented, estimate the average color by selecting it from the color bar. \
                                              Use mouse click to select. \n \n"
                                              f"Ready? \n"
                                              f"Press any key to start this session :)",
                                         pos=(0, 0),
                                         color=self.cfg['text']['color'],
                                         height=self.cfg['text']['height'])

        # Define goodbye message
        texts["goodbye"] = visual.TextStim(self.win,
                                           text=f"Well done! \n \n"
                                                f"You have completed this session. \n"
                                                f"Please wait for further instructions :)",
                                           pos=(0, 0),
                                           color=self.cfg['text']['color'],
                                           height=self.cfg['text']['height'])

        # Define warning message
        texts["warning_error"] = visual.TextStim(self.win,
                                                 text=f"Too large error!",
                                                 pos=(0, 0),
                                                 color='red',
                                                 height=self.cfg['text']['height'])

        texts["warning_rt"] = visual.TextStim(self.win,
                                              text=f"Too slow!",
                                              pos=(0, 0),
                                              color='red',
                                              height=self.cfg['text']['height'])

        # Define practice feedback
        texts["practice_feedback"] = visual.TextStim(self.win,
                                                     text=f"Empty feedback message",
                                                     pos=(0, 0),
                                                     color=self.cfg['text']['color'],
                                                     height=self.cfg['text']['height'])
        # Define escape confirm
        texts["confirm_escape"] = visual.TextStim(self.win,
                                                  text=f"Are you sure to quit (Y/N)? \n",
                                                  pos=(0, 0),
                                                  color=self.cfg['text']['color'],
                                                  height=self.cfg['text']['height'])
        texts['trial_count'] = visual.TextStim(self.win,
                                               pos=(0, -10),
                                               color=self.cfg['text']['color'],
                                               height=self.cfg['text']['height'] * .5)

        return texts


# """
# Run from bash
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='Subject name')
    parser.add_argument('-t', type=int, help='Task type: 0, 1, 2, 3')
    parser.add_argument('-c', help='Configuration file')
    parser.add_argument('-p', help='Parameter file')
    parser.add_argument('-b', type=int, help='Number of blocks')
    parser.add_argument('-r', help='Results folder')
    parser.add_argument('-f', type=bool, help='Practice feedback')

    args = parser.parse_args()
    subject = args.s
    task_type = args.t
    cfg_file = args.c
    par_file = args.p
    block_num = args.b
    res_dir = args.r
    feedback = args.f

    Exp(subject, task_type, cfg_file, par_file, block_num, res_dir, feedback).run_exp()
# """

# TEST
"""
Exp(subject='s1', task_type=2, cfg_file='config/configs.yaml', par_file='config/param/pilot_param_135_315_noise10.yaml',
    block_num=2, res_dir='test_data', feedback=True).run_exp()
"""
