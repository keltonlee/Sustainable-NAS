#pragma once

#include "DSPLib.h"
#include "../cnn/cnn_batchnorm.h"
#include "../cnn/cnn_types.h"
#include "../cnn/cnn_utils.h"
#include "../utils/myuart.h"

CNNLayer_t BatchNorm1[1] = {
{
    .lix = 0,
    .fun = CNN_Intermittent_BatchNormalization, // also verified with CNN_Intermittent_BatchNormalization,
    .weights = (Mat_t){
        .data = 32868,
        .n = 4,
        .ch = 4,
        .h = 1,
        .w = 1
    },
    .bias = (Mat_t){
        .data = 0,
        .n = 0,
        .ch = 0,
        .h = 0,
        .w = 0
    },
    .ifm = (Mat_t){
        .data = 100,
        .n = 1,
        .ch = 4,
        .h = 32,
        .w = 32
    },
    .ofm = (Mat_t){
        .data = 16484,
        .n = 1,
        .ch = 4,
        .h = 32,
        .w = 32
    },
    .parE = (ExeParams_t){
        .Tn = 2,
        .Tm = 0,
        .Tr = 16,
        .Tc = 32,
        .str = 1,
        .pad = 0,
        .lpOdr = OFM_ORIENTED
    },
    .parP = (PreParams_t){
        .preSz = 1,
    },
    .idxBuf = 0
},
};

CNNModel_t network={
    .Layers       = BatchNorm1,
    .numLayers = 1,
    .name = "Verify_BatchNorm"
};

const _q15 INPUTS_DATA[] = {
    16261, 15236,  7821, 31679, 25173,   909, 11652, 28819,  2899,  6937,
     5779,  2047,  4326, 23028, 23428,  9961, 10073, 10826, 17940, 25085,
    20777, 27692, 10778,  7030, 16059, 29336, 14720, 19601, 29374, 18767,
    16059,  9360, 14930, 15088, 30432, 19638, 20719, 11183, 31009, 24378,
    11432, 15555,  3770, 21533, 13163, 19366, 22388, 28285,   731,  3869,
    22710, 14412,  5533, 12468, 22648, 25278,  9630,  2755, 14150, 21680,
    16990, 26439, 20481, 11220, 22861,  5950, 20506,  4812, 26214, 31351,
    32330, 31373,  5276, 12160,   681, 11499,  9249,  6998, 10166, 28724,
    22334, 24255,  5360, 26711, 29989, 18826,  8436, 15724, 13012, 27729,
    19600,  2850, 28644, 23223, 26376, 17702, 13743,   600,  3994,  8463,
    18117, 26746, 27040, 20647, 31219, 13298, 30069,  9871,  1185,  9142,
     8308, 29012,  6069, 26788, 19243,  1601, 12236, 28333,  7972, 11973,
     9997,  1983, 19893, 18967, 30539, 14902,  8788, 17946,  5764, 29838,
    10001, 29844,  8841, 22729,  5416, 11207,  4937, 30187,  2721,  3305,
     1039, 10769, 29458, 27231,  6819,  7345, 14314, 20962, 30467, 30472,
    15929, 27176, 23694, 23213,  8230,  6687, 24324, 32112, 27803,  6760,
    17245,  9540,  7006,  9765,  7984,  5864,  9757, 17548, 19155, 14463,
     3594, 21935,  1086,   957,  2607,  6968,  4545, 22806, 30929, 14841,
     7937, 28468,  3732, 13552, 26721, 20317, 14122, 31333, 25990, 14765,
    11641, 10475,  9117, 24507, 22385, 32261, 15792,  5984, 20766,  6335,
    26862, 32411,  3824,  9632, 32671,    92, 31098,  4060, 22886,   689,
       63, 28454, 18597, 12512, 18004,  5117, 27369, 29767, 15015, 23165,
     6737, 18024, 12587, 10363, 19437, 22676, 27549, 32710,  3681,  4374,
    30217, 18695,  5028, 22359, 18518,  2155,  7920, 14553, 10597, 19203,
    23797, 22951, 28564, 18375, 22972, 27955, 26677,   537,  6678, 23505,
    19326, 19672, 21333, 14990, 11902, 32015, 25378, 15374, 17074, 21098,
    14316,  6108, 30986, 16296, 17009, 10457, 18160, 12987, 20180, 27030,
    13155, 20737, 26548,  9814,  2512,  9084, 32115, 26559,  7604, 32518,
     3758,  9887, 28217, 30547, 10379, 12568, 15807, 24038, 22823, 16732,
    15292, 10281, 29958,  1348, 22763,  1000, 30641, 16220,  5604,  6952,
    30840, 14732, 15277, 32314, 19644, 14912, 11775,  1345,  2136, 13108,
    24956,   740, 17891, 29300, 29717, 15203,  6134, 28475, 26671, 21596,
     1114,  5279, 28599,    77, 30941, 23994,  1741,  1172, 28841,  3532,
    10567, 16473,    40,  2435, 18324, 10888, 19450, 21392,  2468, 27798,
    13623, 16624, 20828, 21088, 13687,  8740,  5644,  2162,  8884,   581,
    23404,  1616, 22684, 10041, 29644, 27635,  6679, 21856,  2368, 13478,
    22390,  1218, 30112,  9216, 24669,   467, 24028, 21544, 28112, 18462,
     5571, 13163, 22510,  4580, 32066, 24219,   168,  2030,  2194, 20538,
     5755, 10071,  5353,  9290, 24564, 13746,  8233,  3252, 19813,  1495,
    15554, 25253,  3603,  5443, 19326,  6540,  6949, 17458, 11575, 14571,
    31797,   934,   854, 18624, 27423, 14081, 14492,  6849,  9240,  5804,
    28814, 25308, 12260,  5026, 28998,  4859,   776, 22823, 24585, 20408,
    16089,  3773, 18762, 18136,  4045,  3967, 20455, 22395,  3746, 25727,
     3753,   304, 15481, 19447, 23018, 17330, 18843, 31305,    78, 24371,
     9674, 12004,  9617,  5894, 26105,  8470, 12078, 20438,  6413, 14867,
    28415, 16448, 31250, 16466, 15906, 30283, 27611, 28879,  8920, 12914,
     2567,    64, 19246, 16495, 12306, 15217,  2789,  9257, 17123, 23631,
    12046, 12346, 18774,  7361, 20921, 21074, 20269,  8058, 15298,   582,
    22813, 24301, 14663, 28813, 17365,   685, 26397, 28165,  8389, 20266,
     3938, 23322, 24136,  3699,  8111, 28649,   667, 24527, 32243, 11283,
     6673,  2375, 19644, 20115, 12282, 26456, 13731, 17217,  8403,   874,
    24870,  5054, 10652,  2612, 26240, 10085,  2955, 16625, 13646, 10567,
    12898, 29294,  9559, 12333, 19886,  2352, 13705,  1836,  5710,  9075,
    22333,  4861, 15543, 30773, 14292, 14828, 28112, 13636, 25919, 13891,
    14699, 23171,  3023,  1511, 16839,   622,  8345, 19884, 14970, 11231,
    28874, 25304, 19699,  8063, 18533, 30028, 26801, 30196, 19683, 30172,
    31903, 23184, 21772,  4935, 26788,  4428, 12512, 11133, 31939, 19025,
     9831,  8611, 15199, 16195, 21084, 19388,  1665,  9652, 12041, 20774,
     8616,  7027,  9615, 20029, 27539, 10947, 18548, 25958, 16277, 29298,
    20346, 25027,  8240,  4639, 14904,  7589,  3828,  4712,  5303, 27631,
     1051, 11329, 13906, 27436,  2555, 16669,  7098, 18552, 13060, 20843,
    25007,  2129, 25369, 16875,  5527,  6335, 25241, 12439, 29447, 20949,
      582,  6449, 22909, 22048, 26604, 13444,  2354, 31661,  3563,  9788,
    28542, 24722, 12920, 29621,   181, 22322,  9740,   435, 15431, 15825,
    13228,  5933,  6154, 27555, 13167,  3054,  2289, 10420,  1681, 13389,
    18007, 23055,  2237, 29151,  8440,  8580, 13820, 30745, 28226, 16231,
    16595,  2120,  3504, 26066,  8941, 15674, 29983, 23901, 22555, 26000,
    26268, 12707,  1637, 23606, 30734, 16176, 15278, 29804, 11586,  7288,
    30792, 15929,  6341, 23973,  9701,  5375,  5499, 27408, 31178, 27222,
    30008, 22023, 22317, 10765,  8234,   529,  1598, 24867, 14366,  4655,
    26750, 12346, 13858, 27352, 14493, 23348, 19202, 14038,  9070, 31120,
     2939, 26445, 29485, 31618, 29595, 28460,  3144,  1473,  4260, 11584,
    18142,  5546, 32498,  6771, 12953,  9609,  9973,   970, 28084,  5935,
     9069, 22651, 20957, 32104, 14131, 24787, 24256, 14936, 21232, 13543,
    22170,  5515, 13348,   478, 12444,  9093,    31,  5830, 12938,  7334,
    27908, 15501,  2882, 23068, 32612, 23861, 25261,  2093, 24650, 10894,
    29392,  1107, 32416,  5807, 27594,   233, 20849,  4977,  4827, 23437,
    15100, 31043, 17114, 24485,  3334, 22317,  4834,  6208,  2064,  3081,
     7364, 20446,  5103,  6265,  6836, 28478, 12227,  2475, 21983, 18764,
    25924, 18754,  6620, 12917,  9624,  9731, 16026, 17594, 15747, 16328,
    17073, 14627, 16152, 13792, 26945, 24633,  5674,  2042,  3999,  5259,
    15940,  1742,  5136, 24692, 32203,  3676,  6870, 25618, 29478,  9522,
    27851,  6457, 28862, 18932, 10494, 29762, 14768,  8678, 30203, 32476,
    16674, 15799, 22308,  9924, 25340, 14421, 18458, 11794, 10934, 31367,
    16262, 24549,  9744,  8915, 13145,  3548, 30401, 31777, 18439, 22863,
    17319,  3806, 12642, 20206,  5321,  2714, 16268, 26711, 30189, 21711,
    18474,  2152, 19103,  6848,  3568,  2772, 25150, 24787,  7796, 23742,
     2154,  7319, 29613, 22285, 19424, 29285,  3087,  1134, 26405, 32248,
    15207,  7247, 14020, 17432, 32591,  1392,  7071, 19885, 22302,  4646,
    29492, 31659, 16847,  8991, 24004, 26096,  2185, 29942, 21050,  2080,
    24500, 13122, 20258, 15859,  4713, 25794, 16745,  1594, 11733, 31943,
    25779, 28899, 10886, 23850, 12802, 13383, 13957,  1716,  6739, 16269,
    16563, 24628, 22640,  6317, 29897, 22633, 20755,  7475, 18429, 29082,
    22622, 18510, 31059,   222, 14564, 29901, 26406, 26460, 14209,  8148,
     6025, 29794, 15081, 17932, 23732,  1140, 20704, 31214,  4802, 18098,
     3811, 31067,  9440, 30370,  1113, 13180, 21202,  4467, 22514, 31740,
    21793, 21759,  7413, 22350, 28675, 23623, 15005, 31305, 11109, 20678,
    20926, 16870, 16410, 32674, 18682,  2498, 24818, 29798, 26946, 11924,
      539, 30440, 18531, 21298, 28229, 14341, 20440, 25121,  2835, 30125,
    14917,    62, 16609,  1115, 18802, 29128, 13598,  7542, 22391,  1686,
     7754, 12879, 27561, 25983, 18549, 11517,   859, 32549, 29932, 31517,
     9559, 20421, 11594, 24111,  1682, 13291,  6656, 26932,  9851, 25405,
    10324, 27528, 26371, 11521,   145,  4145,   860,  4247, 23779, 14486,
     1703,  4756,  8515, 18776, 25870, 24087,  5450, 20645,  6708, 28896,
     6945, 24809, 28358, 17047, 25804,  9864, 21489,   498, 25060, 11443,
    20496,   284, 28959, 30093, 29399, 12684, 22326, 11986,  1207, 25228,
    10912,  5002, 28381,  5465, 11805, 27493, 21697, 20800, 21224, 27478,
     5404,  7428, 29852, 11246, 20370, 29714, 20838, 13108, 11817, 12090,
     8631,  4294,  9807, 12969,  8682,  2701, 14609,  2013,   893, 22545,
     8659,  1708, 19923,  9384, 30710, 17491,  7189, 29427, 30485, 19033,
     1776,  3265, 14636, 25235, 30750, 14119,  1855,  2057,  5744, 16737,
    15381, 19254, 14520, 24804, 17466, 14017, 21077, 29644, 27335, 25994,
    16905, 27479, 16911,  8934,  5359, 23596, 21152, 21265,  3140,  8863,
    20731, 22986, 29443,  4683,  7087, 24614, 19051,  1726,  6221, 10764,
    29976, 17196,  1740, 21656, 10891, 32274, 16255, 13173, 21209, 13271,
    21458, 20795, 12637, 15694, 15219,  6309, 15655,  4477, 32287,  3365,
     6405, 21414, 13258,  2982, 21925, 26771,  5692,  6084, 21564,   230,
    19114,  9449, 16046,  4847, 11770, 18822, 12699, 28519, 14137, 12800,
     6284, 20263, 32259,  7056, 27714, 15362, 15913, 16521,  4187, 19312,
    30696, 11209, 23095, 13892, 18572, 26483, 10874,  4932, 10288,  2890,
     8479,  1074, 28396, 10336, 19327, 27159,  7051,  5855,  7873, 14435,
    16926, 18405, 20158, 10299, 20066, 17277, 19601, 12155, 23737, 15441,
     4219,  7447,  1267, 32640, 19111, 11531, 17187, 10133, 23362,  2319,
    12141, 14029, 22869, 28644, 27006, 20616, 14321, 23074, 21896, 30058,
     2952,  8121, 22950, 28802, 13858, 10791, 26235, 18255, 22074, 23734,
    12715, 10185, 10405, 10316, 15334, 21766, 22603, 20704, 29528, 26692,
    27295, 20895,  3178, 16034,  7829,  6070,  5059, 25379, 16545, 20488,
    17582, 29793, 23158, 17386,  9893,  7605, 17668, 30693, 17322, 12761,
    17752, 10455, 30856,  1633, 18429, 12172, 14484, 20145,  3503, 30500,
     2414, 28060, 17671,  3004,  8936, 13202, 27729, 17567, 31430, 18998,
    31147, 32170, 12784, 21619, 26013,  9037,  9199, 16034, 18579, 32091,
    23758, 29121, 24035, 30922, 18412, 18771,  8413, 31626,  1938,   279,
     2806, 14820, 30028,  2854,  2295, 31205, 25610,   151, 32728,  9245,
    30624, 17059, 26784, 10775, 23553, 25240,  5058, 22580, 10860, 14222,
    22794, 19845, 11035, 31115, 28756, 25416, 24324, 13436, 32761, 31954,
    13207, 19733, 30710,  6127, 19596, 31158, 29077, 25079,   508, 24675,
    12627, 21865,  8285, 15861, 10634, 25027, 22469, 32165, 29836, 24152,
    15576,  6595, 25565, 10255,  8507, 22486,  6524,  3434,  2092, 26731,
    31113, 29497, 26797,  6239, 24300,  7041, 13089,   993, 25315, 26858,
      286, 21624,  6114,   775, 32191, 10168, 21084, 13549, 24356,  8115,
    10640, 16557,  6584, 19642, 29185, 19218,  4426,  4498, 13435, 17708,
    32216, 31119, 22762, 21546, 16452, 14913, 19294, 14559,  6530, 11136,
    23354, 20396, 25812, 30080, 10816, 32597, 12971, 20945, 24372,  5782,
    17506, 19505,  4940, 15489,  7849,  8550, 20084, 10129,  2506, 31999,
     5298, 30759,  5503, 11393,   220, 30409, 14584, 30011,  3226, 15937,
     4522, 15179, 29318, 10704, 32532, 24602, 25248,  6260, 16561, 27903,
    31755, 28107, 10111,  4186, 29509, 15255, 22859,  5464,  1752, 31814,
     1858,   182,  5203, 25727,  8003, 28201, 13736, 30724, 22537, 30474,
     5743, 29342,  2463, 10396, 27760,  5736, 24549,  8359,  3997, 21179,
    17761, 19126,  8390,  9590,  2722, 25204,   555, 24604,   123, 20428,
     7082, 14910, 21562, 12969, 29859, 17802, 24128,  5962, 29798,  5052,
    13108, 30396, 28112,  1741,  1896,  5773, 29034, 22553,  6941,   552,
    30952, 20191, 18125,  9842, 12188, 17807, 16980, 19651, 23592, 27511,
    20577, 22473, 30981,  9542, 19281, 22719, 21804, 22200, 15525,  6892,
    32762,  7360, 25030, 10095, 24881, 19778, 32276,  9098, 26569, 12643,
    30028,  8449, 10649, 27858, 31271,  2584, 24245, 30711, 10854, 19737,
    18266, 30613, 10791, 11619, 12471, 17121, 14571,   614,  7148, 26045,
    21226,  6817,  7190,    14, 23015,  6544,  3776,  6773, 14923,  9316,
    27383, 19312,  5801,  6260, 28031, 16475,  4022, 13542, 14519,  3521,
    32157, 22255,  6902,  9132, 11979,  3124, 29047,  8731, 14169, 14980,
    26861,   151,  6342, 11668, 17601, 17946,  8261,  8007,  8648, 27552,
    19982, 12745, 31441, 23033, 23284,  9439, 23084,  8599,  9907, 16910,
     3946,  1339, 10353,  4557, 32064,  3165, 29362, 19491, 28825, 16687,
     7945, 31426, 10412, 27654,  4160, 29959, 25594,  9748, 13827,  9038,
     7074,  7468, 31974, 21434, 13816,  7643, 19948,   897, 30295, 14965,
    18647, 30178, 17060,   824, 14343,    56,  4796,  1926, 26622,  1742,
    10907, 32512,  8263,  1677, 11936,  8570, 27492,  1168, 13223, 13428,
    20254, 18602, 17951, 12000, 18205, 12051, 31536,  1990, 18639, 20305,
    17261, 13495,  6279, 15370,  6267, 21399, 28552, 16032, 17223,  9731,
    24989, 10464, 24239, 15685,  3569,  6663, 24511, 16305,  1129,  2666,
     1410,    73, 12166, 24651, 13452, 28290, 27268,  7936,  4208, 29618,
     8317, 13051,  9393, 27554, 31569, 24341, 22287, 11003,  8146, 12326,
     4749, 24945,  1584, 12253, 22474,  1033, 23591,  5461, 30290, 12330,
     6482,  2365, 17458, 13051, 13801, 24247,  5464, 10597, 11273, 12687,
    10513, 23265, 15082, 26686, 19961,  8918,  2107, 17462,  3894, 14485,
     1356, 16363, 24523, 11653, 23427,  1592,  1509,   998, 14527, 16745,
      634, 13678,  6500, 12681,   464,  2498, 15966,  7034, 13060, 25937,
     1429, 12592, 27401, 25140, 20733, 17240,   876, 31732, 24164,  4038,
    30002, 10882, 32273, 19108,  9830, 17217,  3036, 27784, 21182, 19593,
    14397, 19620, 17131, 32738, 25275, 32023,  1610, 31401, 31399, 24029,
    29971,  7904,  4727,  3462, 25205,  2442, 23346,  9935, 32669,  1617,
    23148, 12441, 24661,  8494, 26261, 21285,  5569, 30159,  5088, 15916,
    30057, 21886, 28075, 29568, 17264, 32192,  9247, 18287, 24153, 28528,
     6298,  6007,  3246, 16621, 19637, 10377, 11671,  6229, 30263,  2753,
      296, 32382,  5534, 11866, 10002, 29526, 25131, 24439, 19918, 23477,
    14507, 16542,  3519,  7434, 21614,   760, 21606, 29290, 29164,  2206,
    25179, 17875, 20832, 22323, 18666, 27217,  2782,  2003,  5421, 23222,
    11451,  3315,  3681, 25084, 19635,  6615, 11329,  2007,  2562, 31999,
    23575,  7299, 19107, 11725, 32545, 26210,  7586,  9628, 25805, 12970,
    13127,  5321, 14539,  2130, 22245, 26864, 22128, 31283,  3197, 13738,
      310, 16029,  5865, 19543,  2390, 29600, 23856, 20297, 24028,  3577,
      179, 21004,  7103, 22024, 27659, 26871, 24266, 10212,  3750, 17234,
     4818,  9489, 16120, 18078,  8268, 12966, 19747, 15320,  2888, 10319,
    22849,  3652, 24933, 25361,  6214, 12634, 14714, 13271, 18588, 31977,
    28993, 28806, 23438,  4772, 26523, 20122, 30436,  6043, 25449, 13852,
     4054,   125, 16910,  6920,  8238, 17377, 11318,  1381, 21156,  1059,
    12821, 11604, 15858,  7992, 18561,  1211,  5327, 31480, 24505, 13348,
     2594, 23169,  4905, 16651,  1836, 25112, 30134, 15328,  4919, 19189,
    14602,  8335,  9408,  2461,  2655, 27755,  5006,  8760,  7519, 25611,
     2011,  3120, 30880, 12347, 17662,  6252, 31367,  9910, 25992, 21963,
     1207, 26684,  8659, 11839, 27939, 30638, 18022, 32395, 24594,  1382,
    16182,  4440, 26081, 12844, 24042,  2401, 30253, 14863,  5021, 32561,
     7553, 10451, 17161,  2230, 21557, 30371, 31676, 30325, 23088, 13963,
    28080, 20795, 11542, 30275,   145,  1450, 21866, 16464, 13917, 30654,
    11670, 14606, 16536, 11923, 26513,  6827, 10974,  9628, 11838, 10936,
     5619,  3602, 10276, 21663,  4983,  2525, 20508, 23631, 27270, 20031,
    22195, 15801, 11079,   290,  8379, 31995, 15329, 23116, 17832,  5044,
    21191, 12335, 25879, 22665,  6327, 22133, 14753, 31140, 17844, 12372,
    21370, 13535, 26129, 22049, 12432, 29391,  6804, 15936, 22126, 16654,
    21150, 21199,  4515,  4114, 15824,  9718,  6749,  7395, 25392,  5114,
     8067, 22482, 27270, 18470, 31441,  9293, 20730,  5926, 11975, 20324,
    16879,  6227, 16339, 32510,  2280,   629,  8446, 21949, 22907, 11161,
    32739, 14850, 26372,  5436, 32385, 20479, 10432, 29143,  4027,  6944,
      115, 28191,  3101,  2617, 29661, 20271,  3965,  4068, 12538, 29696,
    16304,  2636, 19661, 24273, 12207, 19746, 11778,   949,  5659, 19575,
    26832, 14744, 10507,  9421, 32362, 13724, 19479,   357, 26654,  6436,
     7823, 25259, 11169, 21050, 20014,  7087, 27865, 11270, 12626, 13648,
    24378,  8549,  8444,  2646, 17982, 16193, 18634,  7140,  3187,  5490,
    29855, 23218,  9348, 29944,  5307,  1118,   173, 26751, 17144,  7041,
    27342, 31892, 10342, 28667, 32579, 13482, 32461,  7981, 25927, 21734,
      839,  9565, 21619, 15725,   677, 11783, 20256,  2826, 32528, 23387,
    32155, 23796,  6018,  5636,  8077,  4939, 19525, 18026, 10883, 10469,
    14969, 23075,  1673,  9046, 12932, 15194, 14761,   400, 12724, 19346,
    17507, 16971, 26795,  3885, 20868, 19068, 17167, 23791,  4985, 15165,
      432, 22507, 14488, 28433,  6711, 14328,  2883, 17169, 10798,  2836,
     3056, 13399, 24628, 22937, 18519,  7447,  5781, 27075,   390,  6653,
    31832, 13518, 20746, 14054, 12734, 24106,  1291,  6768, 13442, 22456,
     4983,  5575, 29221,  9830, 12984, 14217, 24618, 15246,   749, 13491,
    30279, 32693, 29616,  6382, 25861, 16461, 29808, 25957, 11413,  3340,
     7826, 30278,  5513, 15189, 19816,  5212, 15164, 24534, 24765, 23560,
    29944, 11035, 21590,  8781, 10885,  7803, 14189, 14682,  1189, 29880,
      930, 31678, 23100, 25051, 28297, 18480, 32333, 13600,  2013,   762,
    11719, 17221, 13349, 31445,  2817, 14651, 17400, 15690,  1522, 20187,
    10988,  8179, 20489, 16966, 10391,  7731, 15143,  6469,  2223, 30433,
     8110, 19419, 17286, 16660, 19695, 16812, 14523, 25433, 22605, 17443,
    30667, 27152, 29414, 10140, 10990, 10024, 29104, 20581, 29047, 17774,
    13931,  3135, 17159,  5785,  1937, 21281,  9022, 30065,  1579, 25262,
    12487,  7720, 31681, 14580, 14524, 27266, 23626,  3656, 16553, 20535,
    23525, 16324, 13684, 26490,  2208, 16407, 21246,  4840, 31555,  2224,
    29876, 15874, 31905,  4795, 18959, 24873, 31176, 31538, 22318, 15730,
     2562, 28636, 26501, 16148, 10201, 25464, 19915, 24368,  5115, 27734,
    23936,  4138, 31899, 22691, 24978, 13099,  9344, 25604, 31719, 23846,
     8903, 27235,  1431, 24419, 24967, 12556, 21990, 31601,  8804, 14301,
    15641, 13142,  8314, 28066, 13441, 20266, 14950, 16378, 10067,  8322,
    14809,  5723, 26234, 13634,  3621,  3679, 31371,  8158, 30042, 25547,
     3013, 28525,  9156, 29955, 25235, 16483, 22195, 29108, 27681, 23049,
    30634, 19110, 19247, 25395, 24646, 27693, 16577, 19437, 18703,  1341,
    28991, 18127, 30324,  4579, 16869, 15184, 18586,   618, 12297,  2505,
     8803, 29584, 15174, 19023, 31883, 16235, 21640, 12754, 20261,  8839,
    20485,  2612,   398, 21927, 15500, 31648, 11719, 10250, 21555, 14637,
     5223,  5876, 20796,  8206, 30748, 22703,  3111,   433, 13678, 19383,
    20466,  1293,  1449,  4651, 26408, 22623, 15354,  5592, 18262,  8904,
    26673, 24065, 14888,  9306, 20640, 28842, 28934,  6903, 21564, 17717,
     2069, 29609, 17905,  8154,  8905, 12445, 22492, 24295,  8978, 27003,
    12392, 19241, 30540, 29390,  9866,  2040,   537,  3681,  1069,  5766,
    18008, 28607,  4041, 19970,   512, 13255, 23484, 26522, 15845, 32076,
     6682, 11120, 22490, 24752, 18734, 24306, 12552,   513, 21613,  6237,
    32288, 19427, 17544, 32062, 16436, 22143,  5761, 28058,  8030, 10440,
    32052, 19452, 20608, 28808,  6856, 23673, 24904,  2815, 29859, 22088,
    16351, 11185,  3350,  6715, 30400, 31995, 12442, 16241,  6014, 32385,
    25296, 20752, 23374, 24568,  9689, 12685, 25687,  6603, 30147, 32095,
    24521,  2287,  5109, 10065, 16377,  3062,  2625,  1751, 21061, 12288,
     8996,  8583,  3530, 25665, 19032,  1178,  3526, 21571, 31469, 29885,
    15365, 14394,  8562,  9029, 14330, 31118, 22243,  4856, 28549, 29249,
    12276, 14612, 28280,  8726, 12830,  5344, 27493,  9288, 28431,  7825,
     8879, 16191,  3686, 24736,  3521, 19783, 18124,  4082,  1015, 16266,
    31791, 24110, 22313, 22709, 14132, 28331,  1620, 27688, 29105, 16347,
       28, 13429, 11338, 23659,  8103, 24430, 29572,  7206,  2550, 18612,
      535, 29160,  9927, 22673, 14022, 32300,  6069, 23128, 13506, 20277,
     7978, 12004, 21693, 31314, 31960,  9079, 22811,   111, 13296, 23387,
    28963, 14585, 29975,  5894, 13943, 19639, 16747, 24985, 15734,  2143,
    14277,  5871, 27603, 24356, 11587, 27623, 11950, 10139,  4672, 22957,
    30746, 10458,  3220, 12370,  5475,  3384, 27728, 25748, 14611, 23082,
    11335, 10988, 15504,  4462,   409, 19874, 23694, 12907, 31420, 21207,
    27585, 23662, 16687, 18057, 13787, 28726,  5420, 19674,  2810, 20088,
     3505, 16058, 24502, 10272,  2624, 11650, 21284,  9257,  2017, 25524,
    22965, 25580, 31210, 22846,  6277, 19423, 27680, 22495, 26928, 30176,
    17724, 27445, 31902, 22553,  2420,  7811, 17805, 29582, 12098, 12577,
     1080,  3181,   992, 15917, 27885, 29555, 22030, 29605,  4236,  8856,
    12033,  3337, 20150, 11998, 12877,  6986, 18764, 16482, 11271, 12481,
     8715, 13311, 24065,  2388, 22088, 19624, 25252,  3439,  1729, 30983,
    19973, 27607, 20115,  6594, 14332, 17148,  5997, 21446, 14764,  3203,
    14612,  3216, 16097, 13678, 18491, 18956,  4183, 15845, 30342, 29881,
    32371, 10559,  8566, 27431,  9317,  2679, 26880, 30414, 25916, 25300,
    14302, 25253,  7199, 14360,  8602, 15028, 16493, 32380,  2116, 20661,
    21992,  3836,  1351,  9537,  1340,  7389, 32384, 30291, 23899, 15532,
    12297, 24587,  3842,  4974, 17202,  4392, 18670, 28324, 20825, 30550,
    13135, 29054, 27520, 16140, 25775, 31226, 30367, 15655, 32307,  6041,
    29671, 11899, 20076, 28041,  4245, 12061, 15916, 27759, 13759, 17204,
     8266,  9864,  6688,  8385, 22492, 18181,  7022,  1143,  8498, 24191,
    20270,  7944, 23151, 28108, 31762, 13123,  9282,  1081,  3258, 28756,
    27497, 32695, 26299, 28436, 28951, 12264,  7889,  6567, 26430, 23059,
    13192, 15754, 16548,  6208, 29389, 11655, 23981,  3775, 12678,  7619,
     1454,  5288, 17875, 13502, 29670, 26039,  4931, 10699, 12127,  8302,
    30330, 25541, 22356, 25908, 14266,  8258, 13609, 21337,  4400, 18904,
    10882, 22901, 21181,  7138,  8689, 12739,  4734, 11270, 26952, 11201,
     3383, 29281,  7470,  8718, 17382,  8340, 13494,  7225, 29373, 15398,
    24648, 31490, 11747, 32031, 18035, 15493, 24095, 21991,  2598,  9357,
    30463,  8044,    92,  1140, 27250,  6546,  8418, 17096,  7789, 10638,
    31495,   120, 14587, 11253,  4734, 16191, 11228, 23517, 25413, 26728,
     3209, 12741, 28111,  6947, 16390, 30202,  9592, 25386, 28711,  7394,
     9148, 11157, 30187, 11260,  4596,  3782, 17912, 22349, 18939,   156,
    20105, 29469, 14093, 24523,  9291, 32056, 17180,  3848, 28751, 19006,
    23490, 24383,  9566, 17074, 23262, 18729,  5002,  9043, 11214, 25288,
    18907, 25641, 17509,  4068, 26204,  6887, 20275, 30711,  1612, 12077,
    16732,    65, 31194,  2690, 19662,  3947, 22277,   633,  9747, 26209,
     4905, 25541, 29813, 16940, 12854, 17497, 14703,  7183, 30598, 29444,
    14287, 12947,  3813, 29421,  1679,  1803, 11595, 24981,  4507,  2124,
    21757, 22940, 16893, 12868,  2030, 12132, 13850, 28882, 25365, 25085,
     8146, 21277, 24912, 24262, 24960, 19473, 26545, 22055, 14917, 12041,
     5938, 15911, 22146, 31296, 32702, 18831, 30183, 16540,  6672, 16456,
     8851, 21916, 32740,  4493,  8320, 16644,   660, 20187, 13091, 18940,
     1786,  4339, 14996, 21847, 26446, 24333,  8189, 28956, 18096,  6622,
    12450, 25814, 17329, 11032, 15911, 18480,  7311,  8613, 23733, 22934,
     9511, 29316, 31541, 11456, 11593,  4846, 28516, 27583,   423, 24749,
    18118, 20983, 17235, 24601, 15351, 28748, 19281, 27968, 20580, 13825,
    16370, 20489, 15715,  3006, 21674, 25315, 14233, 19620, 31929, 29335,
    22017, 23047, 20740, 19611,  3053, 29622, 10386, 21954,  2450, 22734,
     9641,  3572, 16450, 22437,  5901, 11989, 11418, 29422,  5026, 17869,
     8280, 10753, 13745, 18369,  1131, 30849, 13486, 10595, 27358, 24783,
    23672,   141, 15731, 27325,  9380, 12123,  7581, 28068, 29445, 22656,
     9710,  2998,  4887, 30491, 20691, 24917, 16430,  1588,  4501, 31643,
    31111, 24206, 17436,  4873, 32676,   837, 16181, 14979,  6893,   715,
    21965,  6384, 19302,  6123, 22883, 12427, 18319, 14219,  5679, 18825,
     8702, 20959, 23458, 25147, 10723,  3923, 28755, 11627, 20821, 29006,
    17438,   869,  4991, 31200, 24730, 25230, 19087, 17392, 10181, 25575,
    23473, 29720, 19638, 13413,  9927,  8966, 21171, 30409, 29993,  9902,
     8816, 29014, 15305, 14269, 26499, 12772, 23817, 18174,   892, 17016,
    32609,  4954, 10229, 19270, 11375,  3264,  6743, 13173, 25287, 22856,
     6657, 12347, 11697, 21107,  6456,  8988, 13990, 32411,   945,  7066,
    13607,  9704, 19195, 24566, 16282, 22169, 13142, 11518, 10194,  9772,
     2588,  8523, 20224, 24471, 16377, 30816, 17001, 15476, 13654, 19003,
    26769, 28529, 23142, 28205, 13067, 25486,  3550, 31314, 18026, 30021,
    11774, 14075, 10289, 24803, 19756, 18827,  2663, 16999, 23328, 31574,
    23014, 30503, 17623,  5111, 18480,  6760, 20824, 11471,  9822, 31837,
    32762, 18715, 10844, 32394, 10708, 15272, 20668, 32035, 27550, 20746,
    13421,  8730, 30496, 24999, 27090,  7702,  1890,  9643, 17312, 17420,
    20347, 17659, 22572, 21485, 25664, 31206, 23522, 24602, 13828,  7113,
    12363, 23047, 25426,  5273, 23472, 30334, 16941,  3819, 28474, 12279,
    28856, 26177, 17137, 11008, 16736,  1356, 19590, 21875,  5894,  8416,
    16979, 20949, 28826, 27449, 27727,  6826, 21256,  6562,  9466, 17968,
    30323, 29707,  7674,  3944, 20451,  8725, 23523, 11577, 25084, 18975,
     2124,  9953, 26599, 18528, 16646, 30745, 19142, 15982,  8849,  1635,
    12575, 26588, 27197,  6710, 19767, 30299,  1129,   350,  4960,  6154,
    26318, 20516, 16943, 25572, 32593,  3773, 11117, 20761, 19699, 21684,
    17419, 16078, 15292,  5003,  3411,  4928, 32511,  5888, 26394, 21244,
     9382,  8013,  1267, 28276, 15071, 18842, 23400,  2320,  9209, 31438,
    27173,  1252, 14122, 31731,  1368,   277, 19944, 15146, 31187, 20169,
    18572,   335,  4204, 11676, 13289, 10971, 13166, 11294,     2, 10139,
    24874, 31640, 17222, 12899, 11188, 12205, 15898, 22644, 30985,  5979,
    18848, 24629,  9156, 17857, 28938, 23648, 14190,  7391, 32309, 21064,
     5674, 11583,  6663,  9751,  8782, 25038, 15362, 16210, 32372, 20165,
     9819, 18469, 28705,  5140,  1283,  8015, 17590, 31982,  4462, 14918,
    17731, 27468, 31761, 14338,  2480,  5215, 25936,   904, 28753, 23520,
    25184, 27491, 13927, 23116, 25489, 14727,  9550,  4627,  3367,  2220,
     3345,  4869, 18104, 29346, 29854,  2817, 31632,  7568, 24153, 23682,
     7240, 22184, 26252, 19238, 30959, 32632, 18080, 23308, 20690, 17142,
     5266,  9187, 27910, 21097, 23095, 28703,  9365, 22893, 18273, 18476,
    23937, 28766, 21882,  3310,  1859, 11357, 12304, 12260, 15343,  7164,
     8652, 24745, 21848, 15101, 29834, 19986, 21297, 29197, 21876, 13877,
    30094, 17792,  9734,  4937, 32483, 27767, 17847,  6192, 31289, 16262,
    12822, 30515, 27384, 21410, 14982,  6998, 27902, 16953, 25571, 27994,
    14269, 16828,  3343, 10900,  3669,   792, 31212, 16547, 10301,  2060,
    27411,   429, 14544, 17746, 16398, 11058,  7530, 24492,  9682, 30195,
    24759, 27389, 19724,  1674, 22135, 17094,  5539, 24627, 19027, 23290,
    27026,  2595, 20341, 23236,  7719, 18773, 30776,  2240, 19514,  1529,
    22351,  1983,   264,  3413,  2016,  3680, 24229,  3157,  4491, 16665,
    27475, 10702, 23605,  6024, 28681, 18614, 18616, 11140, 17721,  4371,
    24374, 13932, 26057, 20536,    20, 22776, 21970, 26251,  1263, 20182,
    27371, 29740, 26431, 31583, 25629, 30891, 26863, 31162,  2049, 27171,
     1561,  5931, 18111, 11559, 22603, 16332, 10798, 14874,  3590, 19089,
     8560, 30880, 28791, 30687, 27852, 24534, 21525, 11306, 14039, 12039,
    32584,  9945, 18929, 28753,   229,  6005, 26699,  2065,   611, 28456,
    27412, 30406, 27461, 11742, 25127, 25550, 30930, 10247, 13523,  4366,
    26422, 27465,  1779,  3020, 27276, 31459,   655, 29196,  2584, 15693,
    21533, 32547, 28242,  4834, 16189, 17813,   915, 29733,  6032,  9982,
    20044, 10780, 23539, 25713,  5651,  8833, 31255, 23775,  9876,  4865,
    25860, 14339, 19285,  8504, 14072, 29501,  9942,  3552, 18597, 27003,
    13791,  6926,  5933, 20468,   854,  1458, 24117, 24891,   935, 13081,
     3687,  1037, 22730, 22971, 27695,  8709,  9449,  3946,  8900, 18179,
     6416, 16361,  9885, 20839,  9919, 10342,  2660, 14144,  2988, 24308,
    12422, 22570,  3286, 30574, 14799, 21375, 26187, 14212,  8832,  1241,
    24711,  8137, 28046, 21551, 18566,  3795, 25786, 15051,  7434,  6758,
     1924,  5109, 21834, 21114, 14745, 16936, 17120, 20143, 21892, 12354,
     4674, 25207, 32191,  7683, 19909, 13949, 12466,  1454, 31302, 24802,
    26442, 30090,   268,  1787, 15466,  6434,    91, 20788, 10100, 18584,
    18483, 32289, 26410, 29509, 28721,   168,  1784, 14319,  6414,  7878,
    30110,  5645, 19990, 26872, 32748,   533, 27901, 13822, 11910, 19106,
    21435,  7650, 22856,  8163, 19929, 26379, 32722, 17578, 32730, 17482,
    21323, 21617, 27057, 17025, 22791, 11741,  7169, 11407, 22400, 27493,
    17800, 28728,  9680, 14251, 24754, 25472, 20383,  9845, 30833, 19319,
     7651,  5550,    83, 18884, 30374, 21354, 25256, 24856,  9363,  6254,
    11627, 26591, 20129, 19828,   340,  9663, 11637, 14316, 20020, 14588,
    24577, 27643,  7464, 23576, 16654,  8351, 18705,  8444, 25392,  8312,
     5746, 31703, 11947, 14266,  4054,  5160,  6769,  6376, 14993, 23944,
    22665, 22630, 10880,  4615,  7163, 10382, 25966, 21644,  2932, 23105,
    17796, 27827, 22714, 25892, 29172, 18359,  6530, 25951, 19454, 14741,
    10225, 13376, 11115, 26802, 12292, 25325, 27481,  4619, 17547,  7708,
    17279,  5330, 10709,    59, 19459,  2419, 10487, 14824, 11114,  6298,
    12266,  4504, 16442, 16809, 27865,  3690, 13233,  8954, 21289, 20548,
    14991, 20034,  8496, 30384,  3606, 16936,  5058,  9936, 16283, 30019,
    16763, 15457,  2784,  9279, 21632,  4513, 12859, 11901, 16041, 19247,
    25879, 27220, 25975, 21055,  4991, 12241, 30164, 13868,  5340, 32069,
    21186, 28849, 12383,  9876, 21787,  3197, 11892, 16204, 25889, 14185,
    19157, 12502, 10771,  7289, 19104, 24027, 19290, 30770, 24692, 22661,
    29630, 30261,  5097,  7659,  8538, 32500, 15227, 27180, 17659, 31458,
    25547, 15342,  9978, 22872, 30555, 12240,  3785,  3838, 15188, 16053,
    25062,  5270,  6903, 27786, 29475, 20311, 21093, 32455,  7532,  1756,
    25714, 24356, 22524, 26756,  6148, 19250, 21408, 23506,  6876, 27076,
    22781, 20293, 23625,  9532, 25402,  2083
};

const _q15 WEIGHT[] = { 28302, 14649,  9508, 24938 };
const _q15 BIAS[]   = { 17768,  9095, 20218,  5458 };
const _q15 MEAN[]   = { 31611, 30744, 23940, 25051 };
const _q15 VAR[]    = { 18785,  2173,  7455,  2092 };

static void initializeData() {
    memcpy_dma_ext(network.Layers[0].ifm.data, INPUTS_DATA, sizeof(INPUTS_DATA), sizeof(INPUTS_DATA), MEMCPY_WRITE);

    memcpy_dma_ext(network.Layers[0].weights.data + 0*sizeof(WEIGHT), WEIGHT, sizeof(WEIGHT), sizeof(WEIGHT), MEMCPY_WRITE);
    memcpy_dma_ext(network.Layers[0].weights.data + 1*sizeof(WEIGHT), BIAS,   sizeof(BIAS),   sizeof(BIAS),   MEMCPY_WRITE);
    memcpy_dma_ext(network.Layers[0].weights.data + 2*sizeof(WEIGHT), MEAN,   sizeof(MEAN),   sizeof(MEAN),   MEMCPY_WRITE);
    memcpy_dma_ext(network.Layers[0].weights.data + 3*sizeof(WEIGHT), VAR,    sizeof(VAR),    sizeof(VAR),    MEMCPY_WRITE);
}

static void dumpResults() {
    _DBGUART("Layer 0 outputs\r\n");
    CNN_printResult(&network.Layers[0].ofm);
}
