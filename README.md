### To create variables, execute the file execution.sh with the appropriate arguments
### Before executing on a non HPC machine, correct the paths in lines 63 - 71
### Syntax for Different Scenarios

1. **Non-HPC Machine:**
   ```bash
   ./script.sh REG 2014 03 -- 02 -- IBM
HPC Machine:

bash
Copy code
./script.sh HPC 2014 03 -- 02 -- IBM
One Day:

bash
Copy code
./script.sh REG 2014 03 -- 02 -- IBM
One Stock:

bash
Copy code
./script.sh REG 2014 03 -- 02 -- IBM MSFT
Some Days:

bash
Copy code
./script.sh REG 2014 03 -- 02 03 -- IBM
Some Stocks:

bash
Copy code
./script.sh REG 2014 03 -- 02 03 -- IBM A AA ABC MSFT
All Trading Days for Every Stock:

bash
Copy code
./script.sh REG 2014 03 -- all -- IBM A AA ABC MSFT
