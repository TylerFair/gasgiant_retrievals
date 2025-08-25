This directory is made for the fitting of multiwavelength light curves.

The script is as follows:

python cleaned_<trendversion>_freeld.py -c <config>

where <trendversion> is the type of detrending you want to complete
- gp is a gp whitelight fit followed by subtracting it from all wavelength bins and then fitting linear
- linear is a offset + linear with time coefficient
- expramp is to account for exponential ramps in miri data

and; where <config> is the .yaml file that specifies configurations. 
