JWST Flux cube -> Retrieval fitting kit.

- new_version directory contains PLATON atmospheric retrieval code and modified source files for custom stellar template spectra,
    paramterized TP profiles in transmission, Transit Light Source (TLS), and partial cloud coverage. The retrieval is currently slow!!
    And doesn't support multiple instrument offsets, but I'll be improving these shortly.

- light_curve_fitting directory contains the modeling of the whitelight curve and multiwavelength light curves given a fluxcube,
  this currently is supported for the exoTEDRF reduction (https://github.com/radicamc/exoTEDRF), but also has
  manually swappable functions for Eureka (https://github.com/kevin218/Eureka). It's also easy to hack your own fluxcube unpacking
  function following these ones. The full code is controlled by a .YAML file. 

Works In Progress:
- make resolutiuons more intuitive and allow string for native
-  Clean up high-memory demand of fitting script by partitioning runs (which will introduce minor overhead but make it useable for
  small GPU/RAM machines)
- eureka is x1dints unpack exote dis fits
- Make retrieval faster for most high-dimensional cases
- Add native multiple instrument offset support
