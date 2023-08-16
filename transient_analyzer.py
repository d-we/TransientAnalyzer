#! /usr/bin/env python3

import sys
import os
import argparse
import statistics
import subprocess
import numpy
import plotly.graph_objects as go

from plotly.offline import plot
from shutil import copyfile

import userNanoBench
from nanoBench import kernelNanoBench
from common import *

# arch = 'BDW'
TESTRUNS_NORMAL = 10
# TESTRUNS_CBC = 10
TESTRUNS_CBC = 3  # dbg
MEASUREMENTS_PER_TESTRUN_NORMAL = 500
# MEASUREMENTS_PER_TESTRUN_CBC = 5
MEASUREMENTS_PER_TESTRUN_CBC = 1  # dbg
# configfile = "../../configs/cfg_Broadwell_all.txt"
#configfile = "../../configs/cfg_Broadwell_custom.txt"
configfile = ".nanoBench/configs/cfg_Skylake_common.txt"

VERBOSE = True


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    VERBOSE = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_error(message):
    print(f"{bcolors.FAIL}[x] {message}{bcolors.ENDC}")


def log_warning(message):
    print(f"{bcolors.WARNING}[-] {message}{bcolors.ENDC}")


def log_success(message, color=bcolors.OKGREEN):
    print(f"{bcolors.OKGREEN}[+] {message}{bcolors.ENDC}")


def log_info(message, color=bcolors.OKBLUE):
    print(f"{color}[+] {message}{bcolors.ENDC}")


def log_debug(message):
    global VERBOSE
    if VERBOSE:
        print(f"{bcolors.VERBOSE}[*] {message}{bcolors.ENDC}")

def assemble(code, objFile, asmFile='/tmp/asm.s'):
    try:
        if '|' in code:
            code = code.replace('|15',
                                '.byte 0x66,0x66,0x66,0x66,0x66,0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|14',
                                '.byte 0x66,0x66,0x66,0x66,0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|13',
                                '.byte 0x66,0x66,0x66,0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|12',
                                '.byte 0x66,0x66,0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|11',
                                '.byte 0x66,0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|10', '.byte 0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|9', '.byte 0x66,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|8', '.byte 0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00;')
            code = code.replace('|7', '.byte 0x0f,0x1f,0x80,0x00,0x00,0x00,0x00;')
            code = code.replace('|6', '.byte 0x66,0x0f,0x1f,0x44,0x00,0x00;')
            code = code.replace('|5', '.byte 0x0f,0x1f,0x44,0x00,0x00;')
            code = code.replace('|4', '.byte 0x0f,0x1f,0x40,0x00;')
            code = code.replace('|3', '.byte 0x0f,0x1f,0x00;')
            code = code.replace('|2', '.byte 0x66,0x90;')
            code = code.replace('|1', 'nop;')
            code = code.replace('|', '')
        code = '.intel_syntax noprefix;' + code + ';1:;.att_syntax prefix\n'
        with open(asmFile, 'w') as f:
            f.write(code);
        subprocess.check_call(['as', asmFile, '-o', objFile])
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Error (assemble): " + str(e))
        sys.stderr.write(code)
        exit(1)


def objcopy(sourceFile, targetFile):
    try:
        subprocess.check_call(['objcopy', "-j", ".text", '-O', 'binary', sourceFile, targetFile])
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Error (objcopy): " + str(e))
        exit(1)


def createBinaryFile(targetFile, asm=None, objFile=None, binFile=None):
    if asm:
        objFile = '/tmp/tmp.o'
        assemble(asm, objFile)
    if objFile is not None:
        objcopy(objFile, targetFile)
        return True
    if binFile is not None:
        copyfile(binFile, targetFile)
        return True
    return False


def create_tmp_folder():
    # TODO(dwe): randomize this
    foldername = "/tmp/dwe-nanobench"
    try:
        os.mkdir(foldername)
    except FileExistsError:
        pass
    return foldername


def writeHtmlFile(filename, title, head, body, includeDOCTYPE=True):
    with open(filename, 'w') as f:
        if includeDOCTYPE:
            f.write('<!DOCTYPE html>\n')
        f.write('<html>\n'
                '<head>\n'
                '<meta charset="utf-8"/>'
                '<title>' + title + '</title>\n'
                + head +
                '</head>\n'
                '<body>\n'
                + body +
                '</body>\n'
                '</html>\n')


def configurePFCs(arch, events):
    content = ''
    for event in events:
        cfg = get_event_config(arch, event)
        if cfg is not None:
            content += cfg + ' ' + event + '\n'
    kernelNanoBench.setNanoBenchParameters(config=content, fixedCounters=True)


def load_nanobench_configfile(fname, usermode=False):
    if usermode:
        userNanoBench.setNanoBenchParameters(config=fname, fixedCounters=True)
    else:
        content = ""
        with open(fname, "r") as fd:
            for line in fd:
                if not line.startswith("#") and len(line.split(" ")) == 2:
                    content += line
        kernelNanoBench.setNanoBenchParameters(config=content, fixedCounters=True)


def generate_transient_code(maincode, initcode, execute_architecturally):
    transient_code_template = '''
    cpuid; # serialize
    
    # setup xmm4 and xmm5 for divider events
    vxorps xmm4, xmm4, xmm4;
    vxorps xmm5, xmm5, xmm5;
    
    lea rax, [rip+g1];
    clflush [rax];
    lea rax, [rip+g2];
    clflush [rax];
    lea rax, [rip+g3];
    clflush [rax];
    lea rax, [rip+g4];
    clflush [rax];
    # lea rax, [rip+gdbg];
    # mov rdi, [rax];
    mfence;
    call retgadget
    # testcode:
    %s
capture:
    lfence;
    pause;
    jmp capture;
    # global variables (aligned to pagesize)
    .align 4096
g1:
    .space 4096, 0x0;
    .align 4096
g2:
    .space 4096, 0x0;
    .align 4096
g3:
    .space 4096, 0x0;
    .align 4096
g4:
    .space 4096, 0x0;
    .align 4096
g5:
    .space 4096, 0x0;
    .align 4096
g6:
    .space 4096, 0x0;
    .align 4096
g7:
    .space 4096, 0x0;
    .align 4096
g8:
    .space 4096, 0x0;
    .align 4096
gdbg:
    .space 4096, 0x0;
retgadget:
    # initcode:
    %s
    # mark hot section with divider event
    divps xmm4, xmm5
    
    lea r12, [rip+g1];
    add rsp, [r12];
    
    lea r11, [rip+g2];
    add r11, [r12];
    add rsp, [r11];
    
    lea r12, [rip+g3];
    add r12, [r11];
    add rsp, [r12];
    
    lea r11, [rip+g4];
    add r11, [r12];
    add rsp, [r11];
    
    #lea r12, [rip+8197];
    lea r12, [rip+end];
    mov [rsp], r12;
    
    ret;
    .space 8192, 0x0;
end:
    nop;
    # mark hot section with divider event
    mfence;
    divps xmm4, xmm5;
    '''

    architectural_code_template = '''
    cpuid; # serialize
    
    # setup xmm4 and xmm5 for divider events
    vxorps xmm4, xmm4, xmm4;
    vxorps xmm5, xmm5, xmm5;
    
    mfence;
    # initcode:
    %s
    # we clobber R11 and R12 to shadow the behavior of the other code template
    xor r11, r11;
    xor r12, r12;
    
    # mark hot section with divider event 
    divps xmm4, xmm5;
    
    # testcode:
    %s
    
    # mark hot section with divider event
    mfence;
    divps xmm4, xmm5;
    nop;
    '''
    spec_stop = "lfence;"

    if execute_architecturally:
        return architectural_code_template % (initcode, maincode), \
               architectural_code_template % (initcode, spec_stop)
    else:
        return transient_code_template % (maincode, initcode), \
               transient_code_template % ("", initcode)


"""
def dbg_generate_transient_code(maincode, initcode, execute_architecturally):
    log_debug(f"maincode: {maincode}\ninitcode:{initcode}")
    transient_code_template = '''
    cpuid; # serialize
    lea rax, [rip+g1];
    clflush [rax];
    lea rax, [rip+g2];
    clflush [rax];
    lea rax, [rip+g3];
    clflush [rax];
    lea rax, [rip+g4];
    clflush [rax];
    # lea rax, [rip+gdbg];
    # mov rdi, [rax];
    mfence;
    call retgadget
    # testcode:
    %s
capture:
    lfence;
    pause;
    jmp capture;
    # global variables (aligned to pagesize)
    .align 4096
g1:
    .space 4096, 0x0;
    .align 4096
g2:
    .space 4096, 0x0;
    .align 4096
g3:
    .space 4096, 0x0;
    .align 4096
g4:
    .space 4096, 0x0;
    .align 4096
g5:
    .space 4096, 0x0;
    .align 4096
g6:
    .space 4096, 0x0;
    .align 4096
g7:
    .space 4096, 0x0;
    .align 4096
g8:
    .space 4096, 0x0;
    .align 4096
gdbg:
    .space 4096, 0x0;
retgadget:
    # initcode:
    %s
    lea r12, [rip+g1];
    add rsp, [r12];
    
    lea r11, [rip+g2];
    add r11, [r12]
    add rsp, [r11];
    
    lea r12, [rip+g3];
    add r12, [r11];
    add rsp, [r12];
    
    lea r11, [rip+g4];
    add r11, [r12]
    add rsp, [r11];
    
    # create divider event (TODO: test)
    #divps xmm4, xmm5
    
    #lea r12, [rip+8197];
    lea r12, [rip+end];
    mov [rsp], r12;
    ret;
    .space 8192, 0x0;
end:
    nop;
    '''

    architectural_code_template = '''
    cpuid; # serialize
    mfence;
    # initcode:
    %s
    # we clobber R11 and R12 to shadow the behavior of the other code template
    xor r11, r11;
    xor r12, r12;
    # create divider event (TODO: test)
    #divps xmm4, xmm5
    # testcode:
    %s
    nop;
    '''
    spec_stop = "lfence;"

    if execute_architecturally:
        return architectural_code_template % (initcode, maincode), \
               architectural_code_template % (initcode, spec_stop)
    else:
        return transient_code_template % (maincode, initcode), \
               transient_code_template % ("", initcode)
"""


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--asm',
                        type=str,
                        metavar="<intel-assembly>",
                        dest="asm",
                        action="store",
                        help="Assembly code to be measured (executed transiently).",
                        required=False)
    parser.add_argument('--asminit',
                        type=str,
                        metavar="<intel-assembly>",
                        dest="asminit",
                        action="store",
                        help="Preparation routines before transient window. "
                             "R11 and R12 will be clobbered after.",
                        required=False)
    parser.add_argument('--asmfile',
                        type=str,
                        metavar="<assembly file>",
                        dest="asmfile",
                        action="store",
                        help="Assembly code to be measured (stored in a file).",
                        required=False)
    parser.add_argument('--asminitfile',
                        type=str,
                        metavar="<assembly file>",
                        dest="asminitfile",
                        action="store",
                        help="Preparation routines before transient window (stored in a file). "
                             "R11 and R12 will be clobbered after.",
                        required=False)
    parser.add_argument('--cbc',
                        dest="cycle-by-cycle",
                        action="store_true",
                        required=False)
    parser.add_argument('-v', '--verbose',
                        dest="verbose",
                        action="store_true",
                        required=False)
    parser.add_argument('-a', '--architectural',
                        dest="architectural",
                        action="store_true",
                        help="Execute code architecturally (instead of transiently)",
                        required=False)
    parser.add_argument('--nz', '--non-zero',
                        dest="nonzero",
                        action="store_true",
                        help="Only show HPCs that are not 0.0",
                        required=False)
    parser.add_argument('-c', '--clean',
                        dest="clean",
                        action="store_true",
                        help="Only show HPCs with values that are not between -1 and 1",
                        required=False)
    parser.add_argument('-r', '--raw',
                        dest="raw",
                        action="store_true",
                        help="Do not crop to the first and last cycle approximation in cbc mode",
                        required=False)
    parser.add_argument('-u', '--user',
                        dest="usermode",
                        action="store_true",
                        help="Execute measurements in usermode (not as reliable)",
                        required=False)

    return vars(parser.parse_args())


def run_normal_mode(arg_dict, active_code_obj_file, passive_code_obj_file):
    if arg_dict['usermode']:
        userNanoBench.setNanoBenchParameters(
            nMeasurements=MEASUREMENTS_PER_TESTRUN_NORMAL,
            unrollCount=1,
            loopCount=0,
            warmUpCount=10,
            basicMode=True,
            drainFrontend=True)
    else:
        kernelNanoBench.setNanoBenchParameters(
            # 5k measurements yield a very stable results (for non-cbc measurements)
            # nMeasurements=5000,
            nMeasurements=MEASUREMENTS_PER_TESTRUN_NORMAL,
            unrollCount=1,
            loopCount=0,
            warmUpCount=10,
            basicMode=True,
            drainFrontend=True)

    active_results = list()
    passive_results = list()
    for i in range(TESTRUNS_NORMAL):
        log_debug(f"Executing run {i}")
        if arg_dict['usermode']:
            active_results.append(userNanoBench.runNanoBench(codeObjFile=active_code_obj_file))
            passive_results.append(userNanoBench.runNanoBench(codeObjFile=passive_code_obj_file))
        else:
            active_results.append(kernelNanoBench.runNanoBench(codeObjFile=active_code_obj_file))
            passive_results.append(kernelNanoBench.runNanoBench(codeObjFile=passive_code_obj_file))

    log_debug("\t==== ACTIVE ====")
    for key in active_results[0]:
        for testrun in range(TESTRUNS_NORMAL):
            log_debug(f"{key}: {active_results[testrun][key]}")

    log_debug("\t==== PASSIVE ====")
    for key in passive_results[0]:
        for testrun in range(TESTRUNS_NORMAL):
            log_debug(f"{key}: {passive_results[testrun][key]}")

    log_success("\t\t\t\t\t==== DELTA ====")
    for key in active_results[0]:
        deltas = list()
        for testrun in range(TESTRUNS_NORMAL):
            deltas.append(round(active_results[testrun][key] - passive_results[testrun][key], 8))

        delta_min = min(deltas)
        delta_max = max(deltas)
        delta_median = round(statistics.median(deltas), 8)
        delta_range = round(delta_max - delta_min, 8)
        std_deviation = round(numpy.std(deltas), 4)

        if arg_dict['clean']:
            # ignore small values
            if -1 < delta_median < 1:
                continue
        if delta_median:  # highlight non-zero values
            if delta_range > 10:
                color = bcolors.WARNING
            else:
                color = bcolors.OKGREEN
            log_info(
                f"{key:>45}: {delta_median:>6} (rng: {delta_range:>6}, std: {std_deviation:>5})",
                color)
        elif not arg_dict['nonzero']:
            log_info(
                f"{key:>45}: {delta_median:>6} (rng: {delta_range:>6}, std: {std_deviation:>5})",
                bcolors.OKBLUE)


class SlidingWindow:
    def __init__(self, all_values, window_size):
        self.all_values = all_values
        self.all_values_len = len(all_values)
        self.all_values_idx = 0
        self.windowsize = window_size
        self.window = [-1] * window_size

        # initialize window
        self.shift_window(self.windowsize)

    def shift_window(self, shifted_positions=1):
        # returns false when end of values is reached
        if self.all_values_idx + shifted_positions > self.all_values_len:
            return False
        for _ in range(shifted_positions):
            # get current value
            new_value = self.all_values[self.all_values_idx]
            self.all_values_idx += 1

            # pop last element and add first element upfront
            self.window.pop()
            self.window.insert(0, new_value)
        return True

    def get_current_window(self):
        # the internal window is sorted backwards hence we change this
        # to make the interface more intuitive
        return self.window[::-1]

    def get_current_index(self):
        # returns index of last value that was added to the window
        return self.all_values_idx - 1

    def all_same_values(self):
        # returns True if all values in the window are the same
        first_value = self.window[0]
        for v in self.window[1:]:
            if v != first_value:
                return False
        return True


def detect_first_and_last_usercode_cycle(aggregated_nbdict):
    """
    Approach: we detect the plateaus of ARITH.FPU_DIV_ACTIVE
    - these values were injected by the divps instructions of the measurement templates
    - executing one divps instruction either leads to exactly one spike of the counter or two spikes
    - we therefore search for the plateaus between these spikes to detect the boundaries of the usercode
    :param aggregated_nbdict: nbdict after value aggregation
    :return: first_cycle, last_cycle
    """
    divider_pmc = "ARITH.FPU_DIV_ACTIVE"
    sliding_window_size = 5
    divider_trace = aggregated_nbdict[divider_pmc]

    windowsize = 5
    window = SlidingWindow(divider_trace, windowsize)
    plateau = False
    growth_points = list()
    while window.shift_window():
        if window.all_same_values() and plateau is False:
            log_debug(f"Detected plateau: {window.get_current_index() - windowsize}")
            plateau = True
        if plateau is True and not window.all_same_values():
            idx = window.get_current_index()
            log_debug(f"Detected growth after plateau: {idx}")
            growth_points.append(idx)
            plateau = False
    growth_points_len = len(growth_points)
    if growth_points_len != 2 and growth_points_len != 4:
        log_warning("Noisy FPU_DIV_ACTIVE values. First and last cycle could be off!")

    if growth_points_len == 2:
        first_cycle = growth_points[0]
        last_cycle = growth_points[1]
    else:
        first_cycle = growth_points[0]
        last_cycle = growth_points[2]

    # we now further improve the precision by finding the last occurence of
    # BR_MISP_RETIRED.ALL_BRANCHES in this window as it marks the retirement of the branch
    # we used to speculate and hence marks the end of the speculation window with more precision

    br_misp_pmc = "BR_MISP_RETIRED.ALL_BRANCHES"
    br_misp_trace = aggregated_nbdict[br_misp_pmc][first_cycle:last_cycle]
    window = SlidingWindow(br_misp_trace, 2)

    last_changed_idx = -1
    while window.shift_window():
        if window.all_same_values():
            # no change; just go next
            continue
        # found change of the counter
        last_changed_idx = window.get_current_index()

    assert last_changed_idx != -1
    last_cycle = first_cycle + last_changed_idx - 1
    return first_cycle, last_cycle


def aggregate_cycle_by_cycle_results(nbdicts):
    aggregated_nbdict = dict()
    noisy_traces = set()
    for key in nbdicts[0].keys():
        # compute errorchecks on last element as it should hold the sum of all previous
        curr_values = [nbdicts[i][key][-1] for i in range(TESTRUNS_CBC)]
        curr_std_deviation = round(numpy.std(curr_values), 4)
        if curr_std_deviation > 20:
            log_debug(f"Noisy key: {key}")
            noisy_traces.add(key)

        aggregated_value_list = list()
        minimal_end_index = min([len(nbdicts[i][key]) for i in range(TESTRUNS_CBC)])
        for pos in range(minimal_end_index):
            values = [nbdicts[i][key][pos] for i in range(TESTRUNS_CBC)]
            aggregated_value = statistics.median(values)
            aggregated_value_list.append(aggregated_value)
        aggregated_nbdict[key] = aggregated_value_list

    return aggregated_nbdict, noisy_traces


def run_cycle_by_cycle_mode(arg_dict, active_code_obj_file, passive_code_obj_file):
    kernelNanoBench.setNanoBenchParameters(
        nMeasurements=MEASUREMENTS_PER_TESTRUN_CBC,
        unrollCount=1,
        loopCount=0,
        warmUpCount=5,
        basicMode=True,
        drainFrontend=True)
    # we don't get output without endToEnd
    kernelNanoBench.setNanoBenchParameters(endToEnd=True)
    detP23 = True  # TODO(dwe): why and when is this needed?

    log_debug("starting nanoBench")

    nbdicts = list()
    for i in range(TESTRUNS_CBC):
        nbdict = kernelNanoBench.runNanoBenchCycleByCycle(codeObjFile=active_code_obj_file,
                                                          detP23=detP23)
        assert nbdict is not None
        nbdicts.append(nbdict)

    log_debug("nanoBench finished")

    aggregated_nbdict, noisy_traces = aggregate_cycle_by_cycle_results(nbdicts)

    first_cycle, last_cycle = detect_first_and_last_usercode_cycle(aggregated_nbdict)
    log_info(f"usercode from {first_cycle} - {last_cycle}")

    # generate HTML output
    fig = go.Figure()
    fig.update_xaxes(title_text='Cycle')
    for name, values in aggregated_nbdict.items():
        # cut off values based on first/last cycle approximation and rebase them to 0
        if not arg_dict['raw']:
            values_cropped = values[first_cycle:last_cycle]
            values = [v - values_cropped[0] for v in values_cropped]

        if arg_dict['clean'] and max(values) < 1:
            log_debug(f"Dropping {name}")
            continue
        display_name = name
        if name in noisy_traces:
            display_name += " (noisy!)"

        fig.add_trace(go.Scatter(y=values, mode='lines+markers',
                                 line_shape='linear', name=display_name,
                                 marker_size=5, hoverlabel=dict(namelength=-1)))

    config = {'displayModeBar': True,
              'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'lasso2d'],
              'modeBarButtonsToAdd': ['toggleSpikelines', 'hoverclosest', 'hovercompare',
                                      {'name': 'Toggle interpolation mode', 'icon': 'iconJS',
                                       'click': 'interpolationJS'}]}
    body = plot(fig, include_plotlyjs='cdn', output_type='div', config=config)

    body = body.replace('"iconJS"', 'Plotly.Icons.drawline')
    body = body.replace('"interpolationJS"',
                        'function (gd) {Plotly.restyle(gd, "line.shape", gd.data[0].line.shape == "hv" ? "linear" : "hv")}')

    cmdline = ' '.join(('"' + p + '"' if ((' ' in p) or (';' in p)) else p) for p in sys.argv)
    body += '<p><code>sudo ' + cmdline + '</code></p>'

    output_fname = 'graph.html'

    # if DOCTYPE is included, scaling doesn't work properly
    writeHtmlFile(output_fname, 'Graph', '', body, includeDOCTYPE=False)
    os.chown(output_fname, int(os.environ['SUDO_UID']), int(os.environ['SUDO_GID']))
    log_success(f'Output written to {output_fname}')


def main():
    global VERBOSE
    tmpfolder = create_tmp_folder()
    arg_dict = parse_arguments()

    if os.geteuid() != 0:
        print("[-] Script requires root permissions. Aborting!")
        exit(0)

    VERBOSE = arg_dict['verbose'] is True

    kernelNanoBench.resetNanoBench()
    '''
    STATUS QUO:
        xor rax, 1337; * 139 + divps xmm4, xmm5; 
        works often (executed uops: 133)
    '''
    usercode = '''
    #mov rdi, [rip+gdbg];
    xor rax, 1337;
    ''' * 135
    usercode += "divps xmm4, xmm5;"

    # code should count transient cycles in ARITH.FPU_DIV_ACTIVE
    usercode = "divps xmm4, xmm5;" * 500

    # explicitly cached gdbg to check that around 70 load can be performed then
    # usercode = "mov rdi, [rip+gdbg];" * 70
    # usercode += "divps xmm4, xmm5;" * 50

    if arg_dict['asm']:
        usercode = arg_dict['asm']
    elif arg_dict['asmfile']:
        with open(arg_dict['asmfile'], "r") as fd:
            usercode = "\n".join(fd.readlines())
    else:
        log_error("Failed to provide code for measurement!")
        exit(1)

    if arg_dict['asminit']:
        initcode = arg_dict['asminit']
    elif arg_dict['asminitfile']:
        with open(arg_dict['asminitfile'], "r") as fd:
            initcode = "\n".join(fd.readlines())
    else:
        initcode = ""

    architectural = arg_dict['architectural'] is True
    if architectural:
        log_info("Executing code architecturally")

    active_code, passive_code = generate_transient_code(usercode, initcode, architectural)

    log_debug(f"Executing code: {active_code}")

    active_code_obj_file = f"{tmpfolder}/active_code.o"
    active_code_asmfile = f"{tmpfolder}/active_asm.s"
    if arg_dict['usermode']:
        createBinaryFile(active_code_obj_file, asm=active_code)
    else:
        kernelNanoBench.assemble(active_code, active_code_obj_file, active_code_asmfile)

    passive_code_obj_file = f"{tmpfolder}/passive_code.o"
    passive_code_asmfile = f"{tmpfolder}/passive_asm.s"
    if arg_dict['usermode']:
        createBinaryFile(passive_code_obj_file, asm=passive_code)
    else:
        kernelNanoBench.assemble(passive_code, passive_code_obj_file, passive_code_asmfile)


    if arg_dict['usermode']:
        userNanoBench.resetNanoBench()
        userNanoBench.setNanoBenchParameters(basicMode=True, drainFrontend=True)
        load_nanobench_configfile(configfile, usermode=True)
    else:
        kernelNanoBench.resetNanoBench()
        kernelNanoBench.setNanoBenchParameters(basicMode=True, drainFrontend=True)
        load_nanobench_configfile(configfile, usermode=False)

    if arg_dict['cycle-by-cycle'] is True:
        run_cycle_by_cycle_mode(arg_dict, active_code_obj_file, passive_code_obj_file)
    else:
        run_normal_mode(arg_dict, active_code_obj_file, passive_code_obj_file)

    log_success("Finish")


if __name__ == "__main__":
    main()
