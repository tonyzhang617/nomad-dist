#!/bin/bash

# stop on error
set -e

# # Step 1: Execute StableLM with NoMAD-Attention
# echo "Executing StableLM with NoMAD-Attention..."
# taskset -c 0-23 ./app/bin/main -m models/stablelm-3b-4e1t.Q4_0.gguf -n 16384 -pi assets/stablelm-3b-dsub1 2> stablelm_nomad.log

# # Step 2: Compare Performance with Original Attention
# echo "Comparing performance with original dot-product attention..."
# taskset -c 0-23 ./app/bin/main -m models/stablelm-3b-4e1t.Q4_0.gguf -n 16384 2> stablelm_attn.log

# # Step 3: Perform Model Inference on a Learning Corpus for StableLM
# echo "Performing model inference on a learning corpus for StableLM..."
# # make directory for storing attention keys
# mkdir -p assets/stablelm-3b-wikitext2-valid-keys
# taskset -c 0-23 ./app/bin/perplexity -m models/stablelm-3b-4e1t.Q4_0.gguf -c 512 -f data/wikitext-2-raw/wiki.valid.raw -psi assets/stablelm-3b-wikitext2-valid-keys

# Step 4: Learn Codebooks via k-means Clustering for StableLM
echo "Learning codebooks via k-means clustering for StableLM..."
# taskset -c 0-23 python learn_codebooks.py --paths assets/stablelm-3b-wikitext2-valid-keys --save_path assets/stablelm-3b-wikitext2-valid-codebooks --range 0 1024 --d_sub 1 --niter 100 --dim 80
taskset -c 0-23 python learn_codebooks.py --paths assets/stablelm-3b-wikitext2-valid-keys --save_path assets/stablelm-3b-wikitext2-valid-codebooks --range 0 1024 --d_sub 1 --niter 100

# Step 5: Test StableLM with NoMAD-Attention
echo "Testing StableLM with NoMAD-Attention..."
taskset -c 0-23 ./app/bin/main -m models/stablelm-3b-4e1t.Q4_0.gguf -n 1024 -pi assets/stablelm-3b-wikitext2-valid-codebooks -p "What does the const keyword mean in C++? Answer: " 2> /dev/null

# Testing StableLM with NoMAD-Attention...
# What does the const keyword mean in C++? Answer:
# It means that a variable is constant.This means you can not change it's value after being assigned.In general, this is used when dealing with data that may be accessed by other threads or processes.

# A: It also serves as a type hint for the compiler so that the variable cannot be mis-used as an int (e.g. if someone tries to use it in an array of integers). You can also use const with pointers and references, but I would say this is less common practice.
# const float x = 3; //float is a pointer type by default, so we need the extra cast because its value cannot be modified
# const int *p = &x; //this line says that x's value may not be changed after it was assigned in the program

# Q: How to make a custom background for android device I have been using black background for my application and want to change the background color of the app according to the lightness.
# For example, if the light is dimmed then I would like my app to display dark color, and when the light is brighter than it should be displayed in bright colors. Please help me with this issue?

# A: You can use the following code in your MainActivity for creating a custom background.
# public class MainActivity extends AppCompatActivity {

# private Button button;

# @Override
# protected void onCreate(Bundle savedInstanceState) {
# super.onCreate(savedInstanceState);
# setContentView(R.layout.activity_main);
# button = findViewById(R.id.btn1);
# Button btn1= (Button) findViewById(R.id.btn2);

# button.setBackgroundResource(R.drawable.background); // background image here
# }

# The above code is just a simple example for you to have an idea of what I was trying to explain about the custom background. You can try it and let me know if anything was not clear, or please provide your own answer on how to do this.
# For more information visit: http://developer.android.com/training/custom-views/index.html
# Q: How to write an HTTP server for android? I'm building a basic website which is going to have some text and image files, but it's just simple enough that there are no images or graphics on the main page itself.I would like this website to be able to load HTML documents, Java programs and JPEG pictures from a folder of the server.
# How can I make an Android application which will run the files off a web server? The code for the server could be something that is just an HTTP server or some kind of FTP Server? I've heard there's one out there called "HTTPS" which stands for "Hypertext Transfer Protocol Secure".But it would be great to learn about any other type of server and how to do so.
# I know the server side must be able to load files off an HTTP server, but what kind of programming language can I use on my app?
# Can someone help me with this problem, or does anyone have any suggestions for where I would find more info about a server that is just like a simple website and how it works?
# Q: How do I set up an SSL (Secure Hypertext Transfer) Server in the application so I could use my web server to host files from FTP servers, HTTP servers or FTP servers?
# My code will run on a computer program running a program.I want to create a server that can run on a program which is used to load up programs for the app.How do you make an application which loads and runs scripts? I need to write an application that runs a program for my FTP servers or HTTP servers or HTTP servers?
# Thanks.Q: How do i set up a server for running a program for FTP servers or any other types of web pages from websites on the app.How do you make a website for programs in some kind of applications, and so forth?

# A: how to write an application which will be used for HTTP server or how do I run programs Q: What is the best way that is what is the most amazing program
# A: How can I make the site.How do you create a website on any kind of web sites.I set up a service in some of websites, and so forth? A: what is used to make an application for anything or website how I am an app where are these kinds.Q: what-up
# What will be a site is the best way up use the service of web is out there! How do it can i or this is using or that be running with it.How set up, and so forth?
# A: What can i am
# What is how I write is to be a website in website-up and to how will aAll steps completed.

echo "All steps completed."
