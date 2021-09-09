FROM erdosproject/pylot

# =========== UPGRADE PIP =========== #

RUN /usr/bin/python3 -m pip install --upgrade pip

# =========== FIX LANE DETECTION IN PYLOT =========== #

WORKDIR /home/erdos/workspace/pylot/dependencies/lanenet
RUN git pull && pip3 install pyyaml
ENV PYTHONPATH "$PYTHONPATH:$PYLOT_HOME/dependencies/lanenet/"

# ============= FIX CARLA UNICODE ISSUE ============= #
# https://carla-scenariorunner.readthedocs.io/en/latest/FAQ/

ENV PYTHONIOENCODING "utf-8"

# ============= INSTALL LEADERBOARD REQUIREMENTS ============= #

ADD ./rsa /home/erdos/rsa
WORKDIR /home/erdos/rsa/leaderboard/
RUN python3 -m pip install -r requirements.txt

# ============= FIX ModuleNotFoundError: No module named 'tkinter' ============= #

RUN sudo apt -y install python3-tk

# ============= FIX PYGAME ISSUE ============= #

RUN pip3 install pygame==1.9.6

# ============= UPDATE TO NEWEST PYLOT VERSION ============= #

WORKDIR /home/erdos/workspace/pylot
RUN git pull
ENV PYLOT_SHA1 fdba02482f7be134a106479c96ad43273f4235d3
RUN git reset --hard $PYLOT_SHA1

# ============= PREPARE ENVIRONMENT FOR SIMULATION ============= #

RUN echo "" > /home/erdos/.bashrc

ENV LEADERBOARD_ROOT "/home/erdos/rsa/leaderboard/"
ENV SCENARIO_RUNNER_ROOT "/home/erdos/workspace/scenario_runner/"
ENV CARLA_ROOT "$CARLA_HOME"
ENV TEAM_CODE_ROOT "/home/erdos/workspace/pylot"

ENV PYTHONPATH "/home/erdos/workspace/pylot/dependencies/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/agents/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/:/home/erdos/workspace/scenario_runner/:/home/erdos/rsa/leaderboard/"
ENV REPETITIONS "10"
ENV DEBUG_CHALLENGE "0"
ENV ROUTES "/home/erdos/rsa/routes/custom_routes.xml"
ENV SCENARIOS "/home/erdos/rsa/scenarios/custom_scenarios.json"
ENV TEAM_AGENT "${PYLOT_HOME}/pylot/simulation/challenge/ERDOSAgent.py"
ENV CHECKPOINT_ENDPOINT "${LEADERBOARD_ROOT}/results.json"
ENV CHALLENGE_TRACK_CODENAME "MAP"
ENV TEAM_CONFIG "/home/erdos/rsa/confs/frcnn.conf"
ENV TRAFFIC "FALSE"

ENV XAUTHORITY "/home/erdos/.pylot.xauth"

WORKDIR /home/erdos/rsa/scripts
CMD [ "/bin/bash" ]
