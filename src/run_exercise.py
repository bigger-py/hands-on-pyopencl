"""
Hacky way to fix relative imports...
"""
if __name__=="__main__":
    exercise: int = 9
    
    if exercise == 1:
        import exercises.e1_platform_information
    elif exercise == 2:
        import exercises.e2_vector_addition
    elif exercise == 3:
        import exercises.e3_simple_api
    elif exercise == 4:
        import exercises.e4_chained_vector_addition
    elif exercise == 5:
        import exercises.e5_more_vector_addition
    elif exercise == 6:
        import exercises.e6_simple_matmul
    elif exercise == 7:
        import exercises.e7_matmul_optimization_pt1
    elif exercise == 8:
        import exercises.e8_matmul_optimization_pt2
    elif exercise == 9:
        import exercises.e9_pi_reduction