# audiotoolz
some various scripts for processing audio in weird ways that DAW's can't. Requires cuda, you can try modifying it for cpu but it will be agonizingly slow.


first, install python if you haven't already. 

next, do the ol' 
  pip install -r requirements.txt

now that's out of the way, start out by running analyze_and_store to convert your music library into a vector database.
After that, test that it worked with plot_database - this will generate a 2d T_sne plot.

as for the others:

 - plot new song:
    plots an input song against the database, and highlights the closest matches in the database. useful when your db is massive so you can see where a song is (i know, there's a better way to do this, but i'm too riddled with adhd to do anything about it)

 - plot custom axes:
    now this is a fun one. Lets you generate axes from two text inputs to plot the database. Make sure to put both text inputs in quotation marks, lets you do fun stuff like have one axis be "would belong in jojo's bizzare adventure" and have the other be "distressingly similar to cbat"

 - plot audio axes:
   lets you plot one axis as similarity to input music, and the other axis as a T_sne representation. This is particularly useful for constructing mixes - you can use the similarity axis to pick the next song, and then use the t_sne axis to see which tracks will add to the mix best.
