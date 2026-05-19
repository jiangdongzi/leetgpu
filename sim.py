
def check_bank_conflicts():
    for layout in ["simple", "z-order"]:
        print(f"Layout: {layout}")
        for warpId in range(1):
            warpX = warpId % 2
            warpY = warpId // 2
            
            banks_A = []
            banks_B = []
            
            for laneId in range(32):
                if layout == "simple":
                    laneY = laneId // 8
                    laneX = laneId % 8
                else:
                    laneY = laneId % 2 + (laneId // 16) * 2
                    laneX = (laneId % 16) // 2
                    
                C_THREAD_Y = warpY * 4 + laneY
                C_THREAD_X = warpX * 8 + laneX
                
                # As load: r = C_THREAD_Y * 4
                r = C_THREAD_Y * 4
                # Float4 load accesses 4 banks
                banks_A.append(list(range(r, r+4)))
                
                # Bs load: c = C_THREAD_X * 4
                c = C_THREAD_X * 4
                banks_B.append(list(range(c, c+4)))
                
            # Check conflicts for A
            # A conflict occurs if two threads access different words in the same bank
            # For 32-bit banks, a float4 accesses 4 banks.
            # We can flatten the bank accesses and check if any bank has multiple DIFFERENT starting addresses.
            def count_conflicts(banks_list):
                bank_to_addr = {}
                conflicts = 0
                for tid, banks in enumerate(banks_list):
                    addr = banks[0] # The word address
                    for b in banks:
                        bank = b % 32
                        if bank in bank_to_addr:
                            if bank_to_addr[bank] != addr:
                                conflicts += 1
                        bank_to_addr[bank] = addr
                return conflicts
                
            print(f"  A conflicts: {count_conflicts(banks_A)}")
            print(f"  B conflicts: {count_conflicts(banks_B)}")
check_bank_conflicts()
