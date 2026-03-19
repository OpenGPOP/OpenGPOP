#pragma once

struct StartTime {
    int Year, Month, Day, Hour, Min;
    double Sec;
};

struct FullPerturbationParameters {
    StartTime StartTimeUTC;
};