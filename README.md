
### To create variables, execute the file `execution.sh` with the appropriate arguments.

### Before executing on a non-HPC machine, correct the paths in lines 63 - 71.

## Syntax for Different Scenarios

1. Non-HPC Machine:
   ./script.sh REG 2014 03 -- 02 -- IBM

2. HPC Machine:
   ./script.sh HPC 2014 03 -- 02 -- IBM

3. Multiple Stocks:
   ./script.sh REG 2014 03 -- 02 -- IBM MSFT

4. Multiple Days:
   ./script.sh REG 2014 03 -- 02 03 -- IBM

5. All Trading Days for Every Stock:
   ./script.sh REG 2014 03 -- all -- IBM A AA ABC MSFT

6. All Trading Stocks for Some Days:
   ./script.sh REG 2014 03 -- 03 04 05 -- all

7. All Trading Stocks for All Trading Days:
   ./script.sh REG 2014 03 -- all -- all
