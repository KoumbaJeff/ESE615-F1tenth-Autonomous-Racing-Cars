# Lab 1: Intro to ROS 2

## Written Questions

### Q1: During this assignment, you've probably ran these two following commands at some point: ```source /opt/ros/foxy/setup.bash``` and ```source install/local_setup.bash```. Functionally what is the difference between the two?

Answer:
source /opt/ros/foxy/setup.bash sets up the ROS 2 environment (installed ROS 2 packages) while
source install/local_setup.bash sets up your local workspace environment so your own packages (the ones you build) are recognized by ROS 2.

### Q2: What does the ```qos_history_depth``` argument control when creating a subscriber or a publisher? How does different ```qos_history_depth``` affect how messages are handled?

Answer: 
qos_history_depth controls how many messages are kept in the buffer if you use the KEEP_LAST history policy.
If the depth is small, there is a risk of dropping older messages. If it’s large you can hold more messages in memory before getting rid of them (dropping them).

### Q3: What is the ```qos_profile``` when creating a subscriber or a publisher? What properties can you control by changing the ```qos_profile```? List them and briefly describe what each does.

Answer:
A QoS (Quality of Service) profile is a set of rules that dictate how data is shared in ROS 2. 
- Reliability: BEST_EFFORT (messages can be dropped) vs. RELIABLE (guarantees delivery).
- Durability: VOLATILE (no message history) vs. TRANSIENT_LOCAL (cache messages for late joiners).
- History: KEEP_ALL (keep every message) or KEEP_LAST (store only a fixed number).
- Depth: How many messages are stored when using KEEP_LAST.
- Deadline, Liveliness, Lifespan: Advanced timing/availability settings.

### Q4: Do you have to call ```colcon build``` again after you've changed a launch file in your package? (Hint: consider two cases: calling ```ros2 launch``` in the directory where the launch file is, and calling it when the launch file is installed with the package.)

Answer: 
If you run the launch file directly by path (e.g., ros2 launch path/to/lab1_launch.py), you don’t need to rebuild because you are using the file directly from the source.
If you launch it by package name (ros2 launch my_pkg lab1_launch.py), you need to rebuild because the file needs to be copied (installed) into the package’s install fold
