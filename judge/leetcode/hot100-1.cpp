#include <bits/stdc++.h>

using namespace std;

class Solution {
private:
    std::unordered_map<int, int> flag;
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            int num = nums[i];
            auto it = flag.find(target - num);
            if (it != flag.end()) {
                return {i, it->second};
            }
            flag[num] = i;
        }
        return {0, 1};
    }
};
