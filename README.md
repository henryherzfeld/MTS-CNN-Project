
<section class="post">
								<header class="major">									
									<h1>Baseball CNN</h1>
									<p>A binary classifier that uses a convolutional neural network to classify dynamic image data from a baseball video game.</p>
								</header>								
								<h2>Background</h2>
								<p>As of this writing, MLB The Show is a baseball simulation video game developed annually by Sony San Diego Studios primarily for 
								the PlayStation 4. It provides one
								of the more convincing simulations of MLB action that exists on the market. It does a good job of recreating the difficulty of
								arguably one of the hardest things in sports: hitting a baseball.</p>
 By examining pitches in-game, it was found that it is merely 15 frames (30 fps from source) from when the ball is released and a decision has to be made from user input. This means that it takes roughly 0.5 seconds from the time the pitch is thrown for it to reach 
								home plate. Not much, right? The challenge comes when trying to figure out the following in those 0.5 seconds:
								</p><ol>
									<li>The box in the center of the screen that disappears once the pitch is thrown and reappears once the pitch reaches home plate
									represents the strike zone. A ball that is inside the box is a strike, and a ball that is outside the zone is a ball. It is 
									generally good practice for a batter not to swing at any balls. So, a successful player should be able to determine from the path
									of a pitch whether it is a ball or a strike in the 0.5 seconds mentioned.</li> 
									<li>When should a player swing? Thanks to the footage obtained for this project, that answer is more clear to a general user. If you look closely at game footage, the intersection of the green pitcher's mound and the brown dirt in front of home plate create a generally consistent swing timing. Generally, if the player swings:</li>
									<ol>
										<li>Right after the shadow of the ball crosses the brown dirt if the pitch is further away horizontally from the batter
										(outside)</li>
										<li>Right as the shadow of the ball crosses the brown dirt if the pitch is going to land in the center of the strike zone</li>
										<li>Right before the shadow of the ball crosses the brown dirt if the pitch is closer horizontally to the batter (inside)</li>
									</ol>
								</ol>
								<p></p>
								<p>Each of the issues described above can each be its own project. We decided to see if we could use a CNN to address the first 
								problem: classifying balls vs. strikes. </p>
								<h2>Methods</h2>
								<h3>Data Collection</h3>
								<p>With any machine learning problem, the most challenging and time-consuming part is often data collection. This applies to deep 
								learning as well. In order for a convolutional neural network (CNN) to perform classification on image data, we must first collect
								the data. To accomplish this, the PS4's built in screen-recording feature was used, where several unique, full, 9-inning
								games were played in which not a single pitch was swung at via user input.</p>
								<h3>Labeling</h3>
								<p>Next comes the biggest challenge, labeling. Image classification problems often have thousands or tens of thousands of images to use 
								as data. Generally, for a neural network model to measure its performance when it makes a classification, it usually checks its 
								prediction against a "ground-truth answer." So, any pitches used as part of our dataset need to be labeled as "ball" or strike 
								for the purposes of measuring the performance of the network. It is unrealistic to go through the video manually, pick out every
								moment where a pitch begins/ends, and label it manually based on feedback from the game. A  <a href="https://github.com/henryherzfeld/template_matching_labeler"> labelling script</a> was written to allow
								us to automate this process for each game recorded.</p>
								<p>Through this process we were able to obtain 1080 pitches. We stacked 15 consecutive frames of the ball traveling as a single pitch instance. 
								So, we have a total of 16,200 images in our dataset. </p>
								<h3>Downsampling</h3>
								<p>The sample footage shown above was recorded in 720p at 30 frames per second. The videos we used for our project were recorded in 1080p at
								30 frames per second. The computational complexity of a CNN makes it unrealistic to expect that a neural network can train on 16200 1920x1080 RGB
								images in a reasonable amount of time. So, it became necessary to downsample our images. </p>
								<p>So, we changed our images from RGB to grayscale and downsampled them from 1920x1080 to 115x110. Some cropping was also performed to eliminate
								extraneous information from the image. Two samples of frames from a pitch are shown below: </p>
	<div class="image"><img src="https://michaelkeller21.github.io/images/ball1_0.png" width="115" height="110" alt=""></div>
								<div class="image"><img src="https://michaelkeller21.github.io/images/ball1_14.png" width="115" height="110" alt=""></div>
								<h3>Frame Differencing</h3>
								<p>Now that we have our downsampled images, we must perform one more transformation before we begin the network training. Since we chose to stack 15 frames together
								to represent a single pitch instance, we subtracted two of the frames from each other to obtain a resulting image showing the difference between the two frames:
<p align="center"> <img width=400 src="https://hherzf.com/static/76042d35ca6f4098a457869b786590e1/8c332/baseball.png"> </p>
 We 
								trained our CNN on these differences, in an attempt to get the network to recognize the motion of the ball. This is known as frame differencing.</p>
								<h3>Experiments</h3>
								<p>We trained a CNN using several different well-known architectures from research. Using each of these architectures, we ran the train and test process 10 times
								using 5 and 10 fold cross validation. The main performance metric we used to compare network architecture performance was accuracy.</p>
								<h2>Results</h2>
								<p>You can view the results of the experiments described above in the "Results" folder of this Github repo.</p>
							</section>
