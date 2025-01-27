#!/bin/bash
# This script is a slightly modified version of BugsInPy/framework/bugsinpy-test.sh

# Read arguments from TestRunner.
work_dir=$1
type=$2

# Delete existing working environment, if already exists.
cd $work_dir
if [ -d "env" ]; then
    echo "Step 0: Deleting existing working environment."
    rm -rf "env"
    echo "Done."
fi

# Create new working environment.
echo "Step 1: Creating working environment."
python -m venv "env"
echo "Done."

# Activate working environment.
echo "Step 2: Activating environment."
source "env/bin/activate"
echo "Done."

# Change to repo snapshot directory.
cd snapshots/$type

# Read PYTHONPATH
echo "Step 3: Read bug-specific Python path."

DONE=false
until $DONE ;do
read || DONE=true
if [[ "$REPLY" == "pythonpath"* ]]; then
   pythonpath_all="$(cut -d'"' -f 2 <<< $REPLY)"
   if [ "$pythonpath_all" != "" ]; then
       if [[ $work_dir != /* ]]; then
           work_dir_py=${work_dir:1}
           temp_folder=":${work_dir_py}/"
       else
           temp_folder=":${work_dir}/"
       fi
       check_py="$(cut -d';' -f 1 <<< $pythonpath_all; )"
       string2="$(cut -d'/' -f 1 <<< $check_py/ )"
       temp_change_py=";$pythonpath_all"
       deli=";$string2"
       pythonpath_set=${temp_change_py//$deli/$temp_folder}
       pythonpath_set="${pythonpath_set:1}"
       pythonpath_set=$(echo "$pythonpath_set" | sed s#//*#/#g)
   fi
fi
done < "bugsinpy_bug.info"
echo "Done."


# Setup working environment.
echo "Step 4: Installing bug-specific requirements."
sh "bugsinpy_setup.sh"
echo "Done."

# Install requirements.
echo "Step 5: Installing general requirements."
if grep -q '[^[:space:]]' "bugsinpy_requirements.txt"; then
    sed -e '/^\s*#.*$/d' -e '/^\s*$/d' bugsinpy_requirements.txt | xargs -I {} pip install {}
fi
echo "Done."

# Add PYTHONPATH
echo "Step 6: Add bug-specific Python path to working environment."
if [ "$pythonpath_set" != "" ]; then
    if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    saveReply=""
    pythonpath_exist="NO"
    should_change="NO"
    DONE=false
    until $DONE ;do
    read || DONE=true
    if [[ "$pythonpath_exist" == "YES" ]]; then
        if [[ "$REPLY" != "export PYTHONPATH"* ]]; then
            should_change="YES"
        fi
        pythonpath_exist="YES1"
    fi
    if [[ "$REPLY" == "PYTHONPATH="* ]]; then
        pythonpath_exist="YES"
        tes='"'
        if [[ "$REPLY" != *"$pythonpath_set:"* ]]; then
            should_change="YES"
            saveReply=$REPLY
            string1="${REPLY%:*}"
            string2="${REPLY##*:}"
            if [[ "$string2" == *"PYTHONPATH"* ]]; then
                echo "$string1:$pythonpath_set:$string2"
            else
                temp="$"
                temp_py="PYTHONPATH"
                temp2=${REPLY%$tes*}
                echo "$temp2:$pythonpath_set:$temp$temp_py$tes"
            fi
        fi
    fi 
    done <  ~/.bashrc 
    if [[ "$pythonpath_exist" == "NO" ]]; then
        should_change="NO"
        temp_path_not_exist='PYTHONPATH="$pythonpath_set:$PYTHONPATH"'
        echo "$temp_path_not_exist" >> ~/.bashrc 
        echo "export PYTHONPATH" >> ~/.bashrc 
        source ~/.bashrc
    fi
    if [[ "$should_change" == "YES" ]]; then
        echo "SHOULD CHANGE"
        sed -i.bak '/PYTHONPATH=/d' ~/.bashrc
        if [[ "$pythonpath_exist" == "YES1" ]]; then
            sed -i.bak '/export PYTHONPATH/d' ~/.bashrc
        fi

        string1="${saveReply%:*}"
        string2="${saveReply##*:}"
        if [[ "$string2" == *"PYTHONPATH"* ]]; then
            string1="${saveReply%%/*}"
            string2="$(cut -d'"' -f 2 <<< $saveReply)"
            delimeter='"'
            echo "$pythonpath_set"
            echo "$string1"
            echo "$string1$pythonpath_set:$string2$delimeter" >> ~/.bashrc
        else
            temp="$"
            temp_py="PYTHONPATH"
            temp2=${saveReply%$tes*}
            echo "$temp2:$pythonpath_set:$temp$temp_py$tes" >> ~/.bashrc
        fi
        echo "export PYTHONPATH" >> ~/.bashrc
        source ~/.bashrc
    fi
    fi
fi
echo "Done."


# Run tests.
echo "Step 7: Running tests."

rm -f "bugsinpy_pass.txt"
rm -f "bugsinpy_fail.txt"
rm -f "bugsinpy_alltest.txt"

run_commands=""
is_done=false
until $is_done; do
    read || is_done=true
    if [ "$REPLY" != "" ]; then
        run_commands+="$REPLY;"
    if [[ "$REPLY" == *"pytest"* || "$REPLY" == *"py.test"* ]]; then
        use_pytest=true
    else
        use_pytest=false
    fi
fi
done < "bugsinpy_run_test.sh"
# Split commands into an array
IFS=';' read -r -a commands_array <<< "$run_commands"

# Run every command in bugsinpy_run_test.sh
passed_commands=""
failed_commands=""
for index in "${!commands_array[@]}"
do
    command_to_run=${commands_array[index]}
    command_to_run=$(echo $command_to_run | sed -e "s/\r//g")
    output=$($command_to_run 2>&1; echo "$?") 
    if [[ ${output##*$'\n'} == *"OK"* || ${output##*$'\n'} == *"pass"* || $output == *"passed"* || $output == *"OK "* ]]; then
        passed_commands+="$command_to_run;"
    else
        failed_commands+="$command_to_run;"
    fi
done
echo "$passed_commands" &>>"bugsinpy_pass.txt"    
echo "$failed_commands" &>>"bugsinpy_fail.txt"
echo "Done."
deactivate
