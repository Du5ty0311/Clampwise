# Further reading

Texts and papers underlying the methods in Clampwise, grouped by topic. Each analysis entry in this guide gives the references for its individual measurements.


*Patch-clamp practice*

| Reference | Topics |
|---|---|
| Sakmann & Neher (eds) 1995, *Single-Channel Recording*, 2nd ed (Plenum; Springer reprint) | Whole-cell recording (Marty & Neher); the amplifier and series-resistance compensation (Sigworth); capacitance measurement (Gillis); fitting and the statistics of records (Colquhoun & Sigworth) |
| Walz (ed) 2007, *Patch-Clamp Analysis: Advanced Techniques*, 2nd ed, Neuromethods 38 (Humana) | Whole-cell and perforated-patch recording, fast drug application, and their analysis |
| Molleman 2003, *Patch Clamping: An Introductory Guide to Patch Clamp Electrophysiology* (Wiley) | The seal, membrane test and recording modes |
| *The Axon Guide* (Molecular Devices) | Membrane test, Rs compensation and its errors, leak subtraction, filtering and sampling, liquid junction potential |

*Cellular biophysics and neural computation*

| Reference | Topics |
|---|---|
| Johnston & Wu 1995, *Foundations of Cellular Neurophysiology* (MIT Press) | Passive membrane and cable properties, space clamp, quantal analysis of transmission |
| Hille 2001, *Ion Channels of Excitable Membranes*, 3rd ed (Sinauer) | Conductance, reversal potential, Boltzmann activation, the GHK equation |
| Koch 1999, *Biophysics of Computation* (Oxford) | Dendritic filtering of synaptic inputs, limits of the space clamp, synaptic integration |
| Dayan & Abbott 2001, *Theoretical Neuroscience* (MIT Press) | Firing rate, interspike-interval statistics, CV, f-I curves |
| Gerstner, Kistler, Naud & Paninski 2014, *Neuronal Dynamics* (Cambridge; free online) | Adaptation, firing patterns, fitting simple neuron models to recordings |
| Izhikevich 2007, *Dynamical Systems in Neuroscience* (MIT Press) | Firing-pattern classes (tonic, adapting, bursting, delayed), excitability types and rheobase |

*Data analysis*

| Reference | Topics |
|---|---|
| Kass, Eden & Brown 2014, *Analysis of Neural Data* (Springer) | Estimation with uncertainty, the bootstrap, regression, point processes for spike and event trains |
| Nylen & Wallisch 2017, *Neural Data Science: A Primer with MATLAB and Python* (Academic Press) | The analysis pipeline (import, preprocessing, analysis, visualization) |
| Cohen 2014, *Analyzing Neural Time Series Data* (MIT Press) | Filtering, spectra, time-frequency analysis |
| Mitra & Bokil 2008, *Observed Brain Dynamics* (Oxford) | Multitaper spectra with confidence intervals |
| Motulsky & Christopoulos 2004, *Fitting Models to Biological Data Using Linear and Nonlinear Regression* (Oxford) | Fitting, weighting, model comparison, confidence intervals of fitted parameters |
| Motulsky, *Intuitive Biostatistics* (Oxford University Press) | Error bars, paired data, choice of statistical test |

*Methods papers*

- Neher 1992, *Methods Enzymol* 207:123; Barry 1994, *J Neurosci Methods* 51:107: liquid junction potential.
- Williams & Mitchell 2008, *Nat Neurosci* 11:790: voltage-clamp errors in central neurons.
- Golowasch et al. 2009, *J Neurophysiol* 102:2161: dependence of capacitance measurements on method in non-isopotential neurons.
- Traynelis 1998, *J Neurosci Methods*: offline series-resistance correction.
- Silver 2003, *J Neurosci Methods* 130:127: multiple-probability fluctuation analysis; Clements 2003, *J Neurosci Methods*: variance-mean analysis; Saviane & Silver 2006, *J Neurosci Methods* 153:250: errors in the variance.
- Sigworth 1980, *J Physiol*; Traynelis et al. 1993, *Neuron*: non-stationary and peak-scaled fluctuation analysis.
- Saravanan, Berman & Sober 2020, *Neurons, Behavior, Data analysis, and Theory* 3(5): the hierarchical bootstrap for nested data.

*Figures and statistics*

- Weissgerber et al. 2015, *PLoS Biol* 13:e1002128, "Beyond bar and line graphs": showing every data point for small samples.
- Lord et al. 2020, *J Cell Biol* 219:e202001064, "SuperPlots": individual measurements with each replicate's mean; statistics on the means.
- Aarts et al. 2014, *Nat Neurosci* 17:491: non-independence of cells, sweeps and events from one animal (pseudoreplication).
- Ho et al. 2019, *Nat Methods* 16:565, "Moving beyond P values": estimation plots and effect sizes with confidence intervals.
- Cumming 2014, *Psychol Sci* 25:7, "The new statistics": the distinct meanings of SEM, SD and the 95% CI.
- Wilson & Nicoll 2001, *Nature* 410:588: endocannabinoids as the retrograde messenger of DSI.
- Zucker & Regehr 2002, *Annu Rev Physiol* 64:355: short-term synaptic plasticity (paired pulses and trains).
- Bliss & Lomo 1973, *J Physiol* 232:331: long-term potentiation.
