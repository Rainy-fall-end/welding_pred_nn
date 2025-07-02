from typing import Optional
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        my_set = set()
        while head:
            if head in my_set:
                return True
            my_set.add(head)
            head = head.next
        return False