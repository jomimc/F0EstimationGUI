# Description
Graphical user interface for estimating f0 from audio files, and for fixing errors.

How does it estimate the fundamental frequency (f0)?
> pYIN
> melodia
> crepe

New algorithms can easily be added.

Which errors?
> Octave errors can be fixed by shifting pitches up or down one or more octaves.
> Noise, artefacts, and other unwanted signals can be deleted easily.
> Low-energy sounds may be captured by adapting thresholds
> High- and Low-pass filters can also be applied.

What makes it useful?
> Binaural pitch resynthesis. You can select excerpts, and have the extracted f0 resynthesised and played to the right (or left?) ear, while the original audio is played through the left (or right?) ear.
> Playback speed can be controlled, and looped.
> Sometimes it can be difficult to carefully control exactly where a correction should be implemented. So it's possible to directly enter instructions in text. For example, the start and end times of a segment that should be ignored due to unwanted signal.
> Contains visual spectrogram as a guide.

# Requirements
See requirements.txt


# Citation

This code was used in the following studies. Please cite one or more of them if you use this tool.


