\* TSP *\
Minimize
sum_cost: 6 best_route_('A',_'C') + 5 best_route_('A',_'W')
 + 8 best_route_('C',_'A') + 10 best_route_('C',_'N')
 + 25 best_route_('C',_'W') + 7 best_route_('N',_'C')
 + 20 best_route_('N',_'W') + 8 best_route_('W',_'A')
 + 26 best_route_('W',_'C') + 19 best_route_('W',_'N')
Subject To
SubtourElimination_A_C: 4 best_route_('A',_'C') + u_A - u_C <= 3
SubtourElimination_A_W: 4 best_route_('A',_'W') + u_A - u_W <= 3
SubtourElimination_C_A: 4 best_route_('C',_'A') - u_A + u_C <= 3
SubtourElimination_C_N: 4 best_route_('C',_'N') + u_C - u_N <= 3
SubtourElimination_C_W: 4 best_route_('C',_'W') + u_C - u_W <= 3
SubtourElimination_N_C: 4 best_route_('N',_'C') - u_C + u_N <= 3
SubtourElimination_N_W: 4 best_route_('N',_'W') + u_N - u_W <= 3
SubtourElimination_W_A: 4 best_route_('W',_'A') - u_A + u_W <= 3
SubtourElimination_W_C: 4 best_route_('W',_'C') - u_C + u_W <= 3
SubtourElimination_W_N: 4 best_route_('W',_'N') - u_N + u_W <= 3
Sum_Arrivel_A: best_route_('C',_'A') + best_route_('W',_'A') = 1
Sum_Arrivel_C: best_route_('A',_'C') + best_route_('N',_'C')
 + best_route_('W',_'C') = 1
Sum_Arrivel_N: best_route_('C',_'N') + best_route_('W',_'N') = 1
Sum_Arrivel_W: best_route_('A',_'W') + best_route_('C',_'W')
 + best_route_('N',_'W') = 1
Sum_departure_A: best_route_('A',_'C') + best_route_('A',_'W') = 1
Sum_departure_C: best_route_('C',_'A') + best_route_('C',_'N')
 + best_route_('C',_'W') = 1
Sum_departure_N: best_route_('N',_'C') + best_route_('N',_'W') = 1
Sum_departure_W: best_route_('W',_'A') + best_route_('W',_'C')
 + best_route_('W',_'N') = 1
Bounds
 0 <= u_A <= 3
 0 <= u_C <= 3
 0 <= u_N <= 3
 0 <= u_W <= 3
Generals
u_A
u_C
u_N
u_W
Binaries
best_route_('A',_'C')
best_route_('A',_'W')
best_route_('C',_'A')
best_route_('C',_'N')
best_route_('C',_'W')
best_route_('N',_'C')
best_route_('N',_'W')
best_route_('W',_'A')
best_route_('W',_'C')
best_route_('W',_'N')
End
