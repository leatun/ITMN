#include <iostream>
#include <cstdint> // Cần thư viện này để có kiểu int16_t và int32_t
#include <iomanip> // Cần để in ra dạng hex

// --- ĐỊNH NGHĨA CÁC HẰNG SỐ ---
const int IN_DIM = 4;
const int OUT_DIM = 3;

// --- HÀM LINEAR ĐÃ ĐƯỢC ĐIỀU CHỈNH CHO SỐ NGUYÊN 16-BIT ---
// Dựa trên hàm gốc của cậu
void linear_fixed_point(
    const int16_t x[], 
    int16_t y[], 
    const int16_t W[], 
    const int16_t b[],
    int in_dim, 
    int out_dim
) {
    for (int i = 0; i < out_dim; ++i) {
        // Sử dụng int32_t cho 'sum' để tránh bị tràn số khi tích lũy
        int32_t sum = 0; 
        
        for (int j = 0; j < in_dim; ++j) {
            // Phép nhân (int16_t * int16_t) sẽ tự động được thăng cấp lên int32_t
            sum += x[j] * W[i * in_dim + j];
        }
        
        // Cộng bias và gán kết quả cuối cùng (ép kiểu về lại int16_t)
        y[i] = static_cast<int16_t>(sum + (b ? b[i] : 0));
    }
}


int main() {
    // --- KHỞI TẠO DỮ LIỆU ĐẦU VÀO ---
    // Dữ liệu giống hệt file x_input.txt
    const int16_t x[IN_DIM] = {1, 2, -1, 0};

    // Dữ liệu giống hệt file W_input.txt, lưu dưới dạng mảng 1D
    const int16_t W[OUT_DIM * IN_DIM] = {
        // Hàng 0
        1, 1, 2, 3,
        // Hàng 1
        0, 3, -1, 1,
        // Hàng 2
        2, -2, 1, 1
    };

    // Dữ liệu giống hệt file b_input.txt
    const int16_t b[OUT_DIM] = {5, -5, 0};

    // Mảng để chứa kết quả đầu ra
    int16_t y[OUT_DIM];

    // --- GỌI HÀM TÍNH TOÁN ---
    std::cout << "--- Bat dau tinh toan ---" << std::endl;
    linear_fixed_point(x, y, W, b, IN_DIM, OUT_DIM);
    std::cout << "--- Tinh toan hoan tat ---" << std::endl << std::endl;

    // --- IN KẾT QUẢ ---
    std::cout << "--- Ket qua mong doi ---" << std::endl;
    std::cout << "y[0] = 6" << std::endl;
    std::cout << "y[1] = 2" << std::endl;
    std::cout << "y[2] = -3" << std::endl << std::endl;

    std::cout << "--- Ket qua tu ham C++ ---" << std::endl;
    for (int i = 0; i < OUT_DIM; ++i) {
        // In ra cả dạng thập phân và thập lục phân (hex) để dễ so sánh với Vivado
        std::cout << "y[" << i << "] = " << y[i] 
                  << " (hex: " 
                  << std::hex << std::setw(4) << std::setfill('0') << (y[i] & 0xFFFF) 
                  << std::dec << ")" << std::endl;
    }

    return 0;
}