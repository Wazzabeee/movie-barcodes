# movie_color_barcode
 Compress every frame of a move in a single color barcode

### Todo

- [ ] Optimize Parellel Processing using cap.grap() just like sequential version
- [ ] Optimize K-means to speed up the process
- [ ] Add small GUI with all options available
- [ ] Add option to modify the barcode's height (current is frame's height)
- [ ] Provide more feedback to the user on any errors that occur
- [ ] Ensure the software can handle various video formats beyond MP4
- [ ] Allow the software to process multiple videos at once
- [ ] Add custom naming option through CLI
- [ ] Add examples to Readme
- [ ] Develop POC on Hugging Face Space

### Done
- [X] Define max workers using cpu.cores()
- [X] Define default behavior as maximum cores for faster generation