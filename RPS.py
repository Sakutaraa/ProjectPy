import random

def player(prev_play, opponent_history=[]):
    # Khởi tạo các biến
    opponent_history.append(prev_play)
    moves = ['R', 'P', 'S']

    # Nếu chưa có đủ dữ liệu, ra nước ngẫu nhiên
    if len(opponent_history) < 10:
        return random.choice(moves)

    # Phân tích lịch sử và đưa ra quyết định
    # ... (bạn hãy thêm logic phân tích và quyết định ở đây)

    # Ví dụ đơn giản: Ra nước thắng nước đi trước
    if prev_play == 'R':
        return 'P'
    elif prev_play == 'P':
        return 'S'
    else:
        return 'R'
