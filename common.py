#! /usr/bin/env python3


def get_event_config(arch, event):
    if event == 'UOPS':
        if arch in ['CON', 'WOL']: return 'A0.00'  # RS_UOPS_DISPATCHED
        if arch in ['NHM', 'WSM', 'SNB']: return 'C2.01'  # UOPS_RETIRED.ANY
        if arch in ['SNB']: return 'C2.01'  # UOPS_RETIRED.ALL
        if arch in [
            'HSW']: return 'B1.02'  # UOPS_EXECUTED.CORE; note: may undercount due to erratum HSD30
        if arch in ['IVB', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX', 'TGL',
                    'RKL']: return 'B1.01'  # UOPS_EXECUTED.THREAD
        if arch in ['ZEN+', 'ZEN2', 'ZEN3']: return '0C1.00'
    if event == 'RETIRE_SLOTS':
        if arch in ['NHM', 'WSM', 'SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL',
                    'ICL', 'CLX', 'TGL', 'RKL']: return 'C2.02'
    if event == 'UOPS_MITE':
        if arch in ['SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX',
                    'TGL', 'RKL']: return '79.04'
    if event == 'UOPS_MITE>0':
        if arch in ['SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX',
                    'TGL', 'RKL']: return '79.04.CMSK=1'
    if event == 'UOPS_MS':
        if arch in ['NHM', 'WSM']: return 'D1.02'
        if arch in ['SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX',
                    'TGL', 'RKL']: return '79.30'
    if event == 'UOPS_PORT0':
        if arch in ['CON', 'WOL']: return 'A1.01.CTR=0'
        if arch in ['NHM', 'WSM']: return 'B1.01'
        if arch in ['SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX',
                    'TGL', 'RKL']: return 'A1.01'
    if event == 'UOPS_PORT1':
        if arch in ['CON', 'WOL']: return 'A1.02.CTR=0'
        if arch in ['NHM', 'WSM']: return 'B1.02'
        if arch in ['SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX',
                    'TGL', 'RKL']: return 'A1.02'
    if event == 'UOPS_PORT2':
        if arch in ['CON', 'WOL']: return 'A1.04.CTR=0'
        if arch in ['NHM', 'WSM']: return 'B1.04'
        if arch in ['SNB', 'IVB']: return 'A1.0C'
        if arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'CLX']: return 'A1.04'
    if event == 'UOPS_PORT3':
        if arch in ['CON', 'WOL']: return 'A1.08.CTR=0'
        if arch in ['NHM', 'WSM']: return 'B1.08'
        if arch in ['SNB', 'IVB']: return 'A1.30'
        if arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'CLX']: return 'A1.08'
    if event == 'UOPS_PORT4':
        if arch in ['CON', 'WOL']: return 'A1.10.CTR=0'
        if arch in ['NHM', 'WSM']: return 'B1.10'
        if arch in ['SNB', 'IVB']: return 'A1.40'
        if arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'CLX']: return 'A1.10'
    if event == 'UOPS_PORT5':
        if arch in ['CON', 'WOL']: return 'A1.20.CTR=0'
        if arch in ['NHM', 'WSM']: return 'B1.20'
        if arch in ['SNB', 'IVB']: return 'A1.80'
        if arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX', 'TGL',
                    'RKL']: return 'A1.20'
    if event == 'UOPS_PORT6':
        if arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'CLX', 'TGL',
                    'RKL']: return 'A1.40'
    if event == 'UOPS_PORT7':
        if arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL', 'CLX']: return 'A1.80'
    if event == 'UOPS_PORT23':
        if arch in ['ICL', 'TGL', 'RKL']: return 'A1.04'
    if event == 'UOPS_PORT49':
        if arch in ['ICL', 'TGL', 'RKL']: return 'A1.10'
    if event == 'UOPS_PORT78':
        if arch in ['ICL', 'TGL', 'RKL']: return 'A1.80'
    if event == 'DIV_CYCLES':
        if arch in ['NHM', 'WSM', 'SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL',
                    'CLX']: return '14.01'  # undocumented on HSW, but seems to work
        if arch in ['ICL', 'TGL', 'RKL']: return '14.09'
        if arch in ['ZEN+', 'ZEN2', 'ZEN3']: return '0D3.00'
    if event == 'ILD_STALL.LCP':
        if arch in ['NHM', 'WSM', 'SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL',
                    'ICL', 'CLX', 'TGL', 'RKL']: return '87.01'
    if event == 'INST_DECODED.DEC0':
        if arch in ['NHM', 'WSM']: return '18.01'
    if event == 'FpuPipeAssignment.Total0':
        if arch in ['ZEN+', 'ZEN2', 'ZEN3']: return '000.01'
    if event == 'FpuPipeAssignment.Total1':
        if arch in ['ZEN+', 'ZEN2', 'ZEN3']: return '000.02'
    if event == 'FpuPipeAssignment.Total2':
        if arch in ['ZEN+', 'ZEN2', 'ZEN3']: return '000.04'
    if event == 'FpuPipeAssignment.Total3':
        if arch in ['ZEN+', 'ZEN2', 'ZEN3']: return '000.08'
    # the following two counters are undocumented so far, but seem to work
    if event == 'FpuPipeAssignment.Total4':
        if arch in ['ZEN3']: return '000.10'
    if event == 'FpuPipeAssignment.Total5':
        if arch in ['ZEN3']: return '000.20'
    if event == 'UOPS_EXECUTED':
        if arch in ['BDW']: return 'B1.01'
    if event == 'UOPS_ISSUED':
        if arch in ['BDW']: return '0E.01'
    if event == 'UOPS_RETIRED':
        if arch in ['BDW']: return 'C2.01'
    return None
