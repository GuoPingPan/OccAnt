screen_name=${1}

screen -S ${screen_name} -L -Logfile log/screen_log/${screen_name}.log
