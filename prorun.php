<?php
$command = escapeshellcmd('python3 age_mask_videocap.py'); // change to correct python file
$output = shell_exec($command);
echo $output;
//shell_exec('E:');
//$output= shell_exec('hw.py');
//print($output);
?>
