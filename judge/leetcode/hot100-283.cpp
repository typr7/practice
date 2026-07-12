#include <bits/stdc++.h>


using namespace std;

class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        /*
        int n = nums.size();
        if (n == 1) {
            return;
        }

        int head = 0;
        int tail = 1;
        while (head < n && tail < n) {
            if (nums[head]) {
                head++;
                if (head >= tail) {
                    tail = head + 1;
                }
                continue;
            }
            if (nums[tail] == 0) {
                tail++;
                continue;
            }
            nums[head] = nums[tail];
            nums[tail] = 0;
            head++;
            tail++;
        }
        */

        int n = nums.size();
        int head = 0;
        int tail = 0;
        while (tail < n) {
            if (nums[tail]) {
                swap(nums[head], nums[tail]);
                head++;
            }
            tail++;
        }
    }
}
