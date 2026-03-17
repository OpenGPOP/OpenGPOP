#pragma once

struct StartTime {
    int Year, Month, Day, Hour, Min;
    double Sec;
};

struct FullPerturbationParameters {
    StartTime StartTimeUTC;
    double Area_drag, Area_solar, mass, CR, CD;
    int n, m;

    bool Sun = true, Moon = true, SRad = true, Drag = true, SolidEarthTides = true,
         OceanTides = true, Relativity = true;
};