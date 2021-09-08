#!/bin/bash

echo " ________  __                                   ______                             __       __                  _______             __  __  __           __ "
echo "/        |/  |                                 /      \                           /  \     /  |                /       \           /  |/  |/  |         /  |"
echo "\$\$\$\$\$\$\$\$/ \$\$ |____    ______   __    __       /\$\$\$\$\$\$  |  ______    ______        \$\$  \   /\$\$ |  ______        \$\$\$\$\$\$\$  |  ______  \$\$ |\$\$ |\$\$/  _______ \$\$/ "
echo "   \$\$ |   \$\$      \  /      \ /  |  /  |      \$\$ \__\$\$/  /      \  /      \       \$\$\$  \ /\$\$\$ | /      \       \$\$ |__\$\$ | /      \ \$\$ |\$\$ |/  |/       \\$/  "
echo "   \$\$ |   \$\$\$\$\$\$\$  |/\$\$\$\$\$\$  |\$\$ |  \$\$ |      \$\$      \ /\$\$\$\$\$\$  |/\$\$\$\$\$\$  |      \$\$\$\$  /\$\$\$\$ |/\$\$\$\$\$\$  |      \$\$    \$\$< /\$\$\$\$\$\$  |\$\$ |\$\$ |\$\$ |\$\$\$\$\$\$\$  |   "
echo "   \$\$ |   \$\$ |  \$\$ |\$\$    \$\$ |\$\$ |  \$\$ |       \$\$\$\$\$\$  |\$\$    \$\$ |\$\$    \$\$ |      \$\$ \$\$ \$\$/\$\$ |\$\$    \$\$ |      \$\$\$\$\$\$\$  |\$\$ |  \$\$ |\$\$ |\$\$ |\$\$ |\$\$ |  \$\$ |   "
echo "   \$\$ |   \$\$ |  \$\$ |\$\$\$\$\$\$\$\$/ \$\$ \__\$\$ |      /  \__\$\$ |\$\$\$\$\$\$\$\$/ \$\$\$\$\$\$\$\$/       \$\$ |\$\$\$/ \$\$ |\$\$\$\$\$\$\$\$/       \$\$ |  \$\$ |\$\$ \__\$\$ |\$\$ |\$\$ |\$\$ |\$\$ |  \$\$ |   "
echo "   \$\$ |   \$\$ |  \$\$ |\$\$       |\$\$    \$\$ |      \$\$    \$\$/ \$\$       |\$\$       |      \$\$ | \$/  \$\$ |\$\$       |      \$\$ |  \$\$ |\$\$    \$\$/ \$\$ |\$\$ |\$\$ |\$\$ |  \$\$ |   "
echo "   \$\$/    \$\$/   \$\$/  \$\$\$\$\$\$\$/  \$\$\$\$\$\$\$ |       \$\$\$\$\$\$/   \$\$\$\$\$\$\$/  \$\$\$\$\$\$\$/       \$\$/      \$\$/  \$\$\$\$\$\$\$/       \$\$/   \$\$/  \$\$\$\$\$\$/  \$\$/ \$\$/ \$\$/ \$\$/   \$\$/    "
echo "                              /  \__\$\$ |                                                                                                                    "
echo "                              \$\$    \$\$/                                                                                                                     "
echo "                               \$\$\$\$\$\$/                                                                                                                      "
echo -e "\n\n"
echo "Hello and welcome to the Pylot Evaluation of the Rolling Shutter Attack!"
echo "This script will walk you through all necessary configurations you have to set, before the simulation can be executed."
echo "The script is interactive and will expect user inputs from you."
echo "Once all the required parameters are set, the simulation will be executed."
echo -e "\n"

validate_inputs () {
	rsa_pattern_path="/home/erdos/rsa/patterns/$frequency/$exposure/${duty_cycle}/${spot_location}/$pattern.png"
	
	if [ ! -f $rsa_pattern_path ]; 
	then
		echo "Sorry, but the pattern was not found!"
		echo "Please start again and pick a different pattern"
		get_attack_parameters
	fi
}

get_pattern () {
	echo "You can specify which pattern should be used"
	echo "Please select one of the following:"
	echo "120 [1]"
        echo "180 [2]"
        echo "240 [3]"
        echo "300 [4]"
        echo "360 [5]"
        echo "420 [6]"
        echo "480 [7]"
        echo "540 [8]"
        echo "600 [9]"

	read -p 'Pattern: ' pattern_selection

        case $pattern_selection in 
                1) echo "You selected pattern: 120"
		   pattern="120"
                   ;;
                2) echo "You selected pattern: 180"
                   pattern="180"
                   ;;
                3) echo "You selected pattern: 240"
                   pattern="240"
		   ;;
                4) echo "You selected pattern: 300"
                   pattern="300"
                   ;;
                5) echo "You selected pattern: 360"
                   pattern="360"
                   ;;
                6) echo "You selected pattern: 420"
                   pattern="420"
                   ;;
                7) echo "You selected pattern: 480"
                   pattern="480"
                   ;;
                8) echo "You selected pattern: 540"
                   pattern="540"
                   ;;
                9) echo "You selected pattern: 600"
                   pattern="600"
                   ;;
                *) echo "Please pick one of the available options"
                   get_pattern
        esac
        echo "------------------------------------------"

}

get_spot_location () {
	echo "Depending on where the laser hits the camera, the bright white spot is placed at a different location"
	echo "Please pick, one of the following locations:"
	echo "bottom-left [1]"
        echo "bottom-right [2]"
        echo "top-left [3]"
        echo "top-right [4]"

	read -p 'Spot Location: ' spot_location_selection

        case $spot_location_selection in 
                1) echo "You selected: bottom-left"
                   spot_location="b_l"
                   ;;
                2) echo "You selected: bottom-right"
                   spot_location="b_r"
                   ;;
                3) echo "You selected: top-left"
                   spot_location="t_l"
                   ;;
               	4) echo "You selected: top-right"
                   spot_location="t_r"
                   ;;
                *) echo "Please pick one of the available options"
                   get_spot_location
        esac
        echo "------------------------------------------"
}

get_duty_cycle () {
        echo "What is the duty cycle you want to use?"
        echo "NOTE: Higher duty cycle means larger patterns."
        echo "10 [1]"
        echo "20 [2]"
	echo "40 [3]"

        read -p 'Duty Cycle: ' duty_cycle_selection

        case $duty_cycle_selection in 
                1) echo "You selected: 10%"
                   duty_cycle="10"
                   ;;
                2) echo "You selected: 20%"
                   duty_cycle="20"
                   ;;
                3) echo "You selected: 40%"
                   duty_cycle="40"
                   ;;
                *) echo "Please pick one of the available options"
                   get_duty_cycle
        esac
        echo "------------------------------------------"
}

get_exposure () {
        echo "What is the exposure time you want to use?"
        echo "200 [1]"

        read -p 'Exposure Time: ' exposure_selection

        case $exposure_selection in 
                1) echo "You selected: 200us"
                   exposure="200"
                   ;;
                *) echo "Please pick one of the available options"
                   get_exposure
        esac
        echo "------------------------------------------"
}

get_frequency () {
        echo "What modulation frequency do you want to use?"
        echo "750 [1]"

        read -p 'Frequency: ' frequency_selection

        case $frequency_selection in 
                1) echo "You selected: 750Hz"
                   frequency="750"
                   ;;
                *) echo "Please pick one of the available options"
                   get_frequency
        esac
        echo "------------------------------------------"
}

get_traffic_situation () {
        echo "Do you want to activate traffic?"
	echo "NOTE: We do not recommend to activate traffic, since it might block the agent"
        echo "No [1]"
        echo "Yes [2]"

        read -p 'Enable Traffic: ' traffic_selection

        case $traffic_selection in 
                1) echo "You selected: No"
                   traffic="False"
                   traffic_state="no_traffic"
                   ;;
                2) echo "You selected: Yes"
                   traffic="True"
                   traffic_state="traffic"
                   ;;
                *) echo "Please pick one of the available options"
                   get_traffic_situation
        esac
	echo "------------------------------------------"
}

get_object_detector () {
        echo "Which object detector would you like to use?"
        echo "FRCNN [1]"
        echo "SSD [2]"

	read -p 'Object Detector: ' obj_detector_selection

        case $obj_detector_selection in 
                1) echo "You selected: FRCNN"
                   obj_detector="frcnn"
                   ;;
                2) echo "You selected: SSD"
                   obj_detector="ssd"
                   ;;
                *) echo "Please pick one of the available options"
                   get_object_detector
        esac
        echo "------------------------------------------"
}

get_repetitions () {
	echo "How often do you want to run each scenario?"
	echo "Note: Please be aware that the execution time will increase substantially with higher numbers of repetitions."

	read -p 'Repetitions: ' repetitions

	if [[ -n ${repetitions//[0-9]/} ]]; 
	then
    		echo "Please only input an integer!"
		get_repetitions
	fi
        echo "------------------------------------------"
}

get_vehicle_speed () {
        echo "Specify the maximum velocity of the vehicle in m/s."
	echo "Default: 6 m/s"
        read -p 'Velocity (m/s): ' speed

        if [[ -n ${speed//[0-9]/} ]]; 
        then
                echo "Please only input an integer!"
                get_vehicle_speed
        fi
        echo "------------------------------------------"
}

get_attack_parameters () {
	echo "To execute the Rolling Shutter Attack, additional parameters are required:"
	get_frequency
	get_exposure
	get_duty_cycle
	get_spot_location
	get_pattern
	validate_inputs
}

get_simulation_parameters () {
	echo "You need to specify certain parameters for the simulation"
	get_object_detector
	get_traffic_situation
	get_repetitions
	get_vehicle_speed
}

get_simulation_type () {
	echo "What type of simulation would you like to execute?"
	echo "Baseline [1] - just a normal simulation without any attack."
	echo "Rolling Shutter Attack [2] - frames are intercepted and a rolling shutter pattern is overlayed before passed to Pylot."

	read -p 'Simulation type: ' simulation_type

	case $simulation_type in 
		1) echo "You selected: Baseline"
	           echo "------------------------------------------"
		   get_simulation_parameters
                   export TEAM_AGENT=/home/erdos/workspace/pylot/pylot/simulation/challenge/ERDOSAgent.py
                   export RECORD_PATH=/home/erdos/rsa/recordings/custom_scenarios/${traffic_state}/$speed/${obj_detector}/baseline/
                   export TEAM_CONFIG=/home/erdos/rsa/configs/${obj_detector}.conf
                   CHECKPOINT_PATH=/home/erdos/rsa/results/custom_scenarios/${traffic_state}/$speed/${obj_detector}
		   export CHECKPOINT_ENDPOINT=$CHECKPOINT_PATH/baseline.json 
		   export TRAFFIC=$traffic
	           mkdir -p $RECORD_PATH
		   mkdir -p $CHECKPOINT_PATH
		   ;;
		2) echo "You selected: Rolling Shutter Attack"
 	 	   echo "------------------------------------------"
                   get_simulation_parameters
		   get_attack_parameters
		   export TEAM_AGENT=/home/erdos/rsa/agents/ERDOSAgentUnderRSAAttack.py
	           export RECORD_PATH=/home/erdos/rsa/recordings/custom_scenarios/${traffic_state}/$speed/${obj_detector}/rsa/$frequency/$exposure/${duty_cycle}/${spot_location}/$pattern/
	           export TEAM_CONFIG=/home/erdos/rsa/configs/${obj_detector}.conf
                   CHECKPOINT_PATH=/home/erdos/rsa/results/custom_scenarios/${traffic_state}/$speed/${obj_detector}
        	   export CHECKPOINT_ENDPOINT=$CHECKPOINT_PATH/${frequency}_${exposure}_${duty_cycle}_${spot_location}_${pattern}.json
		   export RSA_PATTERN=/home/erdos/rsa/patterns/$frequency/$exposure/${duty_cycle}/${spot_location}/$pattern.png
                   export TRAFFIC=$traffic
		   mkdir -p $RECORD_PATH
                   mkdir -p $CHECKPOINT_PATH
                   ;;
		*) echo "Please pick one of the available options"
		   get_simulation_type
	esac
        echo "------------------------------------------"
}

setup_simulation_env () {
	get_simulation_type
	sed -i "s/--target_speed=.*/--target_speed=${speed}/g" $TEAM_CONFIG
	export REPETITIONS=$repetitions
}

run_evaluation () {
	setup_simulation_env
	echo "All the required parameters have been set, starting simulation now..."
	$CARLA_ROOT/CarlaUE4.sh &
	sleep 10
	$LEADERBOARD_ROOT/scripts/run_evaluation.sh
	./cleanup.sh
}

run_evaluation
