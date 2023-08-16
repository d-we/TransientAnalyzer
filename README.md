# TransientAnalyzer
A nanoBench plugin to analyze instructions in transient execution.

## Examples

### Normalmode Example

![Example Usage](./data/analyzer_example01.png)
The above picture shows the performance counter impact of the following code snippet:
```
mov rdi, 14
xor rdi, rdi
mov rax, [r14]
```
The code snippet is never architecturally executed by only executed inside a transient window. Besides the aggregated values over multiple runs, the tool also outputs the "range" of the values (`rng`), i.e., max(values) - min(values) and the standard deviation (`std`).
These metrics aim to help users understand whether reported values stem from the executed code snippet or from system noise.

### Cycle-By-Cycle Example
Transientanalyzer also supports the cycle-by-cycle inspection mode of nanoBench. An example can be seen below:

```
TODO: add picture from cbc example
```

## Installation
As TransientAnalyzer itself does not require additional contraints, the installation is mainly nanoBench itself:
```
sudo apt install python3 python3-plotly
git submodule --init update
cd nanoBench
make kernel
```
Afterward, the nanoBench kernel module needs to be loaded:
```
sudo insmod nanoBench/kernel/nb.ko
```

## Usage
```
TODO: write down USAGE table

```
The tool supports the following command line flags

| Option                       | Description |
|------------------------------|-------------|
| `-FLAG`                      | TODO

## Implementation
```
TODO: write down implementation
```

## Disclaimer
The code is provided as-is. You are responsible for protecting yourself, your property and data, and others from any risks caused by this code. This code may cause unexpected and undesirable behavior to occur on your machine.