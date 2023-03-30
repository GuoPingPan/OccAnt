screen_name=${1}

screen -S ${screen_name} -L -Logfile logs/screen_log/${screen_name}.log
