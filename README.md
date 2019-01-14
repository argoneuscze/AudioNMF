# Exploring usage of non-negative matrix factorization in audio compression

This repository stores the implementation part of my Master's thesis.

The text part is visible [**here**](https://github.com/argoneuscze/thesis).

## TODO

* Experiment with different chunk sizes / data types
* Tinker with Nimfa NMF settings
* Learn about CELT/SILK
* Use MDCT instead of FFT and make sure to use KBD windowing
* Look into [GstPEAQ](https://github.com/HSU-ANT/gstpeaq) for measuring audio quality

## Notes

* Input/Output is a two-channel PCM WAV 16-bit (unsigned)
* Current workflow is WAV -> My custom format (ANMF) -> WAV, goal is for the ANMF
  file to be smaller than the WAV while not noticeably losing quality
* For now tried naively "folding" samples into a matrix
* Time domain
  * NMF on the entire song doesn't seem feasible
  * Tried splitting the song into smaller chunks and running NMF on those, seems
    viable, but file size is actually larger than before
* Frequency domain
  * Tried using FFT, but applying NMF to the frequency domain seems to cause
    way too much distortion
    * Possible error in implementation?