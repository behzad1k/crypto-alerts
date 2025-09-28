Loading 1h data from: ./datasets/Binance_1INCHUSDT_1h (1).csv
Loaded 24672 1h records from 2020-12-25 05:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (12342 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.471 (452) | Sell 0.533 (452) | Overall 0.502
Momentum_5  : Buy 0.461 (5057) | Sell 0.479 (5232) | Overall 0.470
Momentum_10 : Buy 0.467 (4984) | Sell 0.493 (5164) | Overall 0.480
WILLR       : Buy 0.534 (2148) | Sell 0.576 (2270) | Overall 0.555
STOCH_D     : Buy 0.515 (1865) | Sell 0.547 (1991) | Overall 0.531

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.552
  Feature importance: MACD: 0.137 | WILLR: 0.863
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.554
  Feature importance: MACD: 0.225 | WILLR: 0.538 | STOCH_D: 0.237

============================================================
ANALYZING 4H TIMEFRAME (6173 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.506 (241) | Sell 0.473 (241) | Overall 0.490
Momentum_5  : Buy 0.477 (2723) | Sell 0.492 (2780) | Overall 0.485
Momentum_10 : Buy 0.482 (2553) | Sell 0.495 (2812) | Overall 0.489
WILLR       : Buy 0.529 (1180) | Sell 0.545 (1113) | Overall 0.537
STOCH_D     : Buy 0.491 (1048) | Sell 0.529 (977) | Overall 0.510

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.552
  Feature importance: WILLR: 0.649 | STOCH_D: 0.351
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.521
  Feature importance: MACD: 0.210 | WILLR: 0.537 | STOCH_D: 0.253

============================================================
ANALYZING 6H TIMEFRAME (4116 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.424 (158) | Sell 0.481 (158) | Overall 0.453
Momentum_5  : Buy 0.475 (1814) | Sell 0.498 (1921) | Overall 0.487
Momentum_10 : Buy 0.478 (1740) | Sell 0.500 (1921) | Overall 0.489
WILLR       : Buy 0.489 (820) | Sell 0.529 (712) | Overall 0.508
STOCH_D     : Buy 0.499 (728) | Sell 0.522 (605) | Overall 0.509

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('Momentum_5', 'STOCH_D') -> 0.518
  Feature importance: Momentum_5: 0.501 | STOCH_D: 0.499
Best 3-combo: ('MACD', 'Momentum_5', 'Momentum_10') -> 0.526
  Feature importance: MACD: 0.365 | Momentum_5: 0.338 | Momentum_10: 0.297

============================================================
ANALYZING 8H TIMEFRAME (3087 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.431 (116) | Sell 0.453 (117) | Overall 0.442
Momentum_5  : Buy 0.486 (1357) | Sell 0.510 (1486) | Overall 0.498
Momentum_10 : Buy 0.483 (1329) | Sell 0.504 (1455) | Overall 0.494
WILLR       : Buy 0.513 (604) | Sell 0.566 (505) | Overall 0.537
STOCH_D     : Buy 0.500 (570) | Sell 0.544 (419) | Overall 0.519

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'STOCH_D') -> 0.555
  Feature importance: MACD: 0.539 | STOCH_D: 0.461
Best 3-combo: ('Momentum_5', 'Momentum_10', 'WILLR') -> 0.552
  Feature importance: Momentum_5: 0.350 | Momentum_10: 0.246 | WILLR: 0.404

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.514 accuracy (1864 signals)
3 timeframes agree: 0.446 accuracy (936 signals)
Overall collision accuracy: 0.491 (2800 signals)
Collision precision: 0.492
Collision recall: 0.491
Collision rate: 0.454

Analysis completed successfully!

 
Loading 1h data from: ./datasets/Binance_ATOMUSDT_1h (1).csv
Loaded 39182 1h records from 2019-04-29 04:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (19599 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.480 (748) | Sell 0.447 (748) | Overall 0.463
Momentum_5  : Buy 0.476 (8227) | Sell 0.474 (8308) | Overall 0.475
Momentum_10 : Buy 0.479 (8011) | Sell 0.479 (8190) | Overall 0.479
WILLR       : Buy 0.551 (3273) | Sell 0.553 (3654) | Overall 0.552
STOCH_D     : Buy 0.531 (2818) | Sell 0.541 (3247) | Overall 0.536

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'STOCH_D') -> 0.538
  Feature importance: MACD: 0.413 | STOCH_D: 0.587
Best 3-combo: ('MACD', 'Momentum_5', 'STOCH_D') -> 0.532
  Feature importance: MACD: 0.308 | Momentum_5: 0.386 | STOCH_D: 0.306

============================================================
ANALYZING 4H TIMEFRAME (9805 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.484 (386) | Sell 0.460 (387) | Overall 0.472
Momentum_5  : Buy 0.482 (4342) | Sell 0.478 (4420) | Overall 0.480
Momentum_10 : Buy 0.487 (4258) | Sell 0.487 (4340) | Overall 0.487
WILLR       : Buy 0.552 (1656) | Sell 0.536 (1795) | Overall 0.544
STOCH_D     : Buy 0.541 (1465) | Sell 0.526 (1552) | Overall 0.533

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.545
  Feature importance: WILLR: 0.613 | STOCH_D: 0.387
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.532
  Feature importance: MACD: 0.254 | WILLR: 0.480 | STOCH_D: 0.265

============================================================
ANALYZING 6H TIMEFRAME (6539 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.488 (244) | Sell 0.514 (245) | Overall 0.501
Momentum_5  : Buy 0.462 (2959) | Sell 0.481 (3008) | Overall 0.471
Momentum_10 : Buy 0.471 (2907) | Sell 0.491 (2944) | Overall 0.481
WILLR       : Buy 0.538 (1126) | Sell 0.576 (1230) | Overall 0.558
STOCH_D     : Buy 0.511 (990) | Sell 0.577 (1060) | Overall 0.545

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.580
  Feature importance: WILLR: 0.625 | STOCH_D: 0.375
Best 3-combo: ('MACD', 'Momentum_5', 'Momentum_10') -> 0.540
  Feature importance: MACD: 0.278 | Momentum_5: 0.433 | Momentum_10: 0.289

============================================================
ANALYZING 8H TIMEFRAME (4905 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.462 (195) | Sell 0.508 (195) | Overall 0.485
Momentum_5  : Buy 0.483 (2242) | Sell 0.493 (2294) | Overall 0.488
Momentum_10 : Buy 0.487 (2192) | Sell 0.499 (2293) | Overall 0.493
WILLR       : Buy 0.544 (855) | Sell 0.549 (885) | Overall 0.547
STOCH_D     : Buy 0.544 (746) | Sell 0.539 (800) | Overall 0.541

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.559
  Feature importance: WILLR: 0.506 | STOCH_D: 0.494
Best 3-combo: ('MACD', 'Momentum_10', 'WILLR') -> 0.535
  Feature importance: MACD: 0.278 | Momentum_10: 0.246 | WILLR: 0.476

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.491 accuracy (2707 signals)
3 timeframes agree: 0.461 accuracy (1617 signals)
Overall collision accuracy: 0.480 (4324 signals)
Collision precision: 0.478
Collision recall: 0.480
Collision rate: 0.441

Analysis completed successfully!

 
Loading 1h data from: ./datasets/Binance_AVAXUSDT_1h (1).csv
Loaded 26926 1h records from 2020-09-22 06:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (13468 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.463 (497) | Sell 0.488 (496) | Overall 0.475
Momentum_5  : Buy 0.466 (5449) | Sell 0.491 (5851) | Overall 0.479
Momentum_10 : Buy 0.479 (5430) | Sell 0.501 (5711) | Overall 0.490
WILLR       : Buy 0.538 (2386) | Sell 0.547 (2463) | Overall 0.542
STOCH_D     : Buy 0.510 (2096) | Sell 0.541 (2172) | Overall 0.525

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.551
  Feature importance: WILLR: 0.697 | STOCH_D: 0.303
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.541
  Feature importance: MACD: 0.176 | WILLR: 0.616 | STOCH_D: 0.208

============================================================
ANALYZING 4H TIMEFRAME (6737 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.458 (253) | Sell 0.460 (252) | Overall 0.459
Momentum_5  : Buy 0.476 (2931) | Sell 0.499 (3047) | Overall 0.487
Momentum_10 : Buy 0.481 (2817) | Sell 0.495 (3100) | Overall 0.489
WILLR       : Buy 0.528 (1268) | Sell 0.549 (1186) | Overall 0.538
STOCH_D     : Buy 0.510 (1134) | Sell 0.548 (1044) | Overall 0.528

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'STOCH_D') -> 0.550
  Feature importance: MACD: 0.575 | STOCH_D: 0.425
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.539
  Feature importance: MACD: 0.321 | WILLR: 0.412 | STOCH_D: 0.267

============================================================
ANALYZING 6H TIMEFRAME (4491 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.480 (171) | Sell 0.448 (172) | Overall 0.464
Momentum_5  : Buy 0.471 (1936) | Sell 0.486 (2127) | Overall 0.479
Momentum_10 : Buy 0.482 (1924) | Sell 0.493 (2120) | Overall 0.488
WILLR       : Buy 0.525 (868) | Sell 0.542 (777) | Overall 0.533
STOCH_D     : Buy 0.510 (790) | Sell 0.537 (684) | Overall 0.522

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('Momentum_5', 'STOCH_D') -> 0.529
  Feature importance: Momentum_5: 0.544 | STOCH_D: 0.456
Best 3-combo: ('Momentum_5', 'WILLR', 'STOCH_D') -> 0.532
  Feature importance: Momentum_5: 0.427 | WILLR: 0.288 | STOCH_D: 0.285

============================================================
ANALYZING 8H TIMEFRAME (3369 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.554 (112) | Sell 0.460 (113) | Overall 0.507
Momentum_5  : Buy 0.486 (1468) | Sell 0.502 (1621) | Overall 0.495
Momentum_10 : Buy 0.495 (1476) | Sell 0.509 (1609) | Overall 0.502
WILLR       : Buy 0.525 (615) | Sell 0.529 (573) | Overall 0.527
STOCH_D     : Buy 0.513 (554) | Sell 0.530 (504) | Overall 0.521

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('Momentum_10', 'STOCH_D') -> 0.544
  Feature importance: Momentum_10: 0.445 | STOCH_D: 0.555
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.537
  Feature importance: MACD: 0.333 | WILLR: 0.342 | STOCH_D: 0.324

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.509 accuracy (2052 signals)
3 timeframes agree: 0.418 accuracy (1236 signals)
Overall collision accuracy: 0.475 (3288 signals)
Collision precision: 0.476
Collision recall: 0.475
Collision rate: 0.488

Analysis completed successfully!

 
Loading 1h data from: ./datasets/Binance_BNBUSDT_1h (1).csv
Loaded 52052 1h records from 2017-11-06 03:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (26038 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.472 (980) | Sell 0.444 (980) | Overall 0.458
Momentum_5  : Buy 0.477 (10217) | Sell 0.466 (9729) | Overall 0.472
Momentum_10 : Buy 0.493 (10150) | Sell 0.482 (9397) | Overall 0.488
WILLR       : Buy 0.558 (3730) | Sell 0.552 (5405) | Overall 0.554
STOCH_D     : Buy 0.537 (3098) | Sell 0.529 (4938) | Overall 0.532

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.554
  Feature importance: WILLR: 0.834 | STOCH_D: 0.166
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.547
  Feature importance: MACD: 0.121 | WILLR: 0.684 | STOCH_D: 0.195

============================================================
ANALYZING 4H TIMEFRAME (13028 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.477 (484) | Sell 0.439 (485) | Overall 0.458
Momentum_5  : Buy 0.499 (5631) | Sell 0.466 (5299) | Overall 0.483
Momentum_10 : Buy 0.497 (5581) | Sell 0.469 (5129) | Overall 0.483
WILLR       : Buy 0.561 (1845) | Sell 0.531 (2722) | Overall 0.543
STOCH_D     : Buy 0.543 (1569) | Sell 0.504 (2487) | Overall 0.519

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.552
  Feature importance: WILLR: 0.725 | STOCH_D: 0.275
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.538
  Feature importance: MACD: 0.171 | WILLR: 0.581 | STOCH_D: 0.247

============================================================
ANALYZING 6H TIMEFRAME (8690 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.447 (318) | Sell 0.453 (318) | Overall 0.450
Momentum_5  : Buy 0.482 (3898) | Sell 0.454 (3616) | Overall 0.468
Momentum_10 : Buy 0.514 (3897) | Sell 0.494 (3555) | Overall 0.504
WILLR       : Buy 0.542 (1272) | Sell 0.526 (1831) | Overall 0.533
STOCH_D     : Buy 0.518 (1104) | Sell 0.502 (1722) | Overall 0.508

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'Momentum_5') -> 0.544
  Feature importance: MACD: 0.373 | Momentum_5: 0.627
Best 3-combo: ('Momentum_5', 'Momentum_10', 'WILLR') -> 0.541
  Feature importance: Momentum_5: 0.469 | Momentum_10: 0.215 | WILLR: 0.316

============================================================
ANALYZING 8H TIMEFRAME (6519 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.558 (233) | Sell 0.502 (233) | Overall 0.530
Momentum_5  : Buy 0.502 (2992) | Sell 0.474 (2780) | Overall 0.489
Momentum_10 : Buy 0.511 (2996) | Sell 0.478 (2780) | Overall 0.495
WILLR       : Buy 0.548 (961) | Sell 0.493 (1388) | Overall 0.516
STOCH_D     : Buy 0.541 (841) | Sell 0.495 (1273) | Overall 0.513

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.521
  Feature importance: MACD: 0.457 | WILLR: 0.543
Best 3-combo: ('Momentum_5', 'WILLR', 'STOCH_D') -> 0.505
  Feature importance: Momentum_5: 0.288 | WILLR: 0.336 | STOCH_D: 0.376

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.496 accuracy (4152 signals)
3 timeframes agree: 0.450 accuracy (2546 signals)
Overall collision accuracy: 0.478 (6698 signals)
Collision precision: 0.479
Collision recall: 0.478
Collision rate: 0.514

Analysis completed successfully!

 
Loading 1h data from: ./datasets/Binance_BTCUSDT_1h (1).csv
Loaded 53988 1h records from 2017-08-17 04:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (27006 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.467 (973) | Sell 0.463 (972) | Overall 0.465
Momentum_5  : Buy 0.479 (9474) | Sell 0.449 (8725) | Overall 0.465
Momentum_10 : Buy 0.499 (9512) | Sell 0.470 (8683) | Overall 0.485
WILLR       : Buy 0.582 (3632) | Sell 0.542 (5506) | Overall 0.558
STOCH_D     : Buy 0.549 (2926) | Sell 0.515 (4931) | Overall 0.527

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.558
  Feature importance: WILLR: 0.849 | STOCH_D: 0.151
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.553
  Feature importance: MACD: 0.141 | WILLR: 0.690 | STOCH_D: 0.169

============================================================
ANALYZING 4H TIMEFRAME (13512 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.503 (501) | Sell 0.421 (501) | Overall 0.462
Momentum_5  : Buy 0.499 (5447) | Sell 0.466 (5023) | Overall 0.483
Momentum_10 : Buy 0.501 (5450) | Sell 0.472 (4919) | Overall 0.487
WILLR       : Buy 0.559 (1988) | Sell 0.527 (2951) | Overall 0.540
STOCH_D     : Buy 0.551 (1652) | Sell 0.495 (2687) | Overall 0.516

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.545
  Feature importance: MACD: 0.250 | WILLR: 0.750
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.559
  Feature importance: MACD: 0.266 | WILLR: 0.488 | STOCH_D: 0.246

============================================================
ANALYZING 6H TIMEFRAME (9014 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.497 (350) | Sell 0.424 (349) | Overall 0.461
Momentum_5  : Buy 0.492 (3855) | Sell 0.453 (3500) | Overall 0.474
Momentum_10 : Buy 0.508 (3900) | Sell 0.477 (3480) | Overall 0.494
WILLR       : Buy 0.561 (1388) | Sell 0.508 (2032) | Overall 0.530
STOCH_D     : Buy 0.530 (1177) | Sell 0.480 (1855) | Overall 0.500

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.545
  Feature importance: MACD: 0.372 | WILLR: 0.628
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.536
  Feature importance: MACD: 0.234 | WILLR: 0.441 | STOCH_D: 0.325

============================================================
ANALYZING 8H TIMEFRAME (6762 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.525 (244) | Sell 0.494 (243) | Overall 0.509
Momentum_5  : Buy 0.498 (2979) | Sell 0.473 (2706) | Overall 0.486
Momentum_10 : Buy 0.508 (3062) | Sell 0.476 (2690) | Overall 0.493
WILLR       : Buy 0.562 (1097) | Sell 0.499 (1576) | Overall 0.525
STOCH_D     : Buy 0.525 (925) | Sell 0.481 (1444) | Overall 0.499

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.549
  Feature importance: WILLR: 0.641 | STOCH_D: 0.359
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.548
  Feature importance: MACD: 0.288 | WILLR: 0.376 | STOCH_D: 0.336

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.493 accuracy (4285 signals)
3 timeframes agree: 0.444 accuracy (2897 signals)
Overall collision accuracy: 0.473 (7182 signals)
Collision precision: 0.473
Collision recall: 0.473
Collision rate: 0.532

Analysis completed successfully!

 
Loading 1h data from: ./datasets/Binance_LTCUSDT_1h (1).csv
Loaded 51164 1h records from 2017-12-13 03:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (25594 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.454 (975) | Sell 0.485 (974) | Overall 0.469
Momentum_5  : Buy 0.457 (10224) | Sell 0.467 (10127) | Overall 0.462
Momentum_10 : Buy 0.473 (10262) | Sell 0.483 (10020) | Overall 0.478
WILLR       : Buy 0.562 (3633) | Sell 0.561 (4753) | Overall 0.562
STOCH_D     : Buy 0.549 (2953) | Sell 0.548 (4070) | Overall 0.549

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.563
  Feature importance: WILLR: 0.730 | STOCH_D: 0.270
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.553
  Feature importance: MACD: 0.179 | WILLR: 0.547 | STOCH_D: 0.274

============================================================
ANALYZING 4H TIMEFRAME (12806 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.493 (489) | Sell 0.462 (489) | Overall 0.478
Momentum_5  : Buy 0.475 (5591) | Sell 0.475 (5501) | Overall 0.475
Momentum_10 : Buy 0.479 (5431) | Sell 0.474 (5418) | Overall 0.476
WILLR       : Buy 0.568 (1925) | Sell 0.557 (2368) | Overall 0.562
STOCH_D     : Buy 0.551 (1632) | Sell 0.542 (2078) | Overall 0.546

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.552
  Feature importance: WILLR: 0.663 | STOCH_D: 0.337
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.548
  Feature importance: MACD: 0.182 | WILLR: 0.575 | STOCH_D: 0.243

============================================================
ANALYZING 6H TIMEFRAME (8542 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.483 (319) | Sell 0.503 (320) | Overall 0.493
Momentum_5  : Buy 0.469 (3815) | Sell 0.464 (3763) | Overall 0.466
Momentum_10 : Buy 0.485 (3744) | Sell 0.482 (3803) | Overall 0.483
WILLR       : Buy 0.560 (1341) | Sell 0.553 (1567) | Overall 0.556
STOCH_D     : Buy 0.535 (1151) | Sell 0.542 (1383) | Overall 0.539

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.559
  Feature importance: MACD: 0.119 | WILLR: 0.881
Best 3-combo: ('Momentum_10', 'WILLR', 'STOCH_D') -> 0.536
  Feature importance: Momentum_10: 0.299 | WILLR: 0.410 | STOCH_D: 0.291

============================================================
ANALYZING 8H TIMEFRAME (6408 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.530 (232) | Sell 0.496 (232) | Overall 0.513
Momentum_5  : Buy 0.486 (2902) | Sell 0.478 (2903) | Overall 0.482
Momentum_10 : Buy 0.481 (2831) | Sell 0.476 (2921) | Overall 0.479
WILLR       : Buy 0.564 (1025) | Sell 0.540 (1162) | Overall 0.551
STOCH_D     : Buy 0.541 (871) | Sell 0.571 (1062) | Overall 0.557

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'STOCH_D') -> 0.584
  Feature importance: MACD: 0.261 | STOCH_D: 0.739
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.551
  Feature importance: MACD: 0.198 | WILLR: 0.481 | STOCH_D: 0.320

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.496 accuracy (4261 signals)
3 timeframes agree: 0.491 accuracy (2671 signals)
Overall collision accuracy: 0.494 (6932 signals)
Collision precision: 0.495
Collision recall: 0.494
Collision rate: 0.541

Analysis completed successfully!

 
Loading 1h data from: ./datasets/Binance_SHIBUSDT_1h (1).csv
Loaded 21413 1h records from 2021-05-10 11:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (10709 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.384 (432) | Sell 0.455 (431) | Overall 0.419
Momentum_5  : Buy 0.436 (3944) | Sell 0.486 (4460) | Overall 0.463
Momentum_10 : Buy 0.460 (3855) | Sell 0.507 (4447) | Overall 0.485
WILLR       : Buy 0.552 (1524) | Sell 0.603 (1208) | Overall 0.574
STOCH_D     : Buy 0.529 (1226) | Sell 0.574 (917) | Overall 0.548

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.580
  Feature importance: WILLR: 0.743 | STOCH_D: 0.257
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.575
  Feature importance: MACD: 0.282 | WILLR: 0.543 | STOCH_D: 0.175

============================================================
ANALYZING 4H TIMEFRAME (5356 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.388 (201) | Sell 0.460 (202) | Overall 0.424
Momentum_5  : Buy 0.442 (2170) | Sell 0.506 (2456) | Overall 0.476
Momentum_10 : Buy 0.449 (1968) | Sell 0.509 (2472) | Overall 0.482
WILLR       : Buy 0.561 (795) | Sell 0.584 (517) | Overall 0.570
STOCH_D     : Buy 0.532 (656) | Sell 0.536 (386) | Overall 0.534

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.583
  Feature importance: WILLR: 0.712 | STOCH_D: 0.288
Best 3-combo: ('Momentum_5', 'WILLR', 'STOCH_D') -> 0.549
  Feature importance: Momentum_5: 0.203 | WILLR: 0.551 | STOCH_D: 0.246

============================================================
ANALYZING 6H TIMEFRAME (3571 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.461 (141) | Sell 0.423 (142) | Overall 0.442
Momentum_5  : Buy 0.437 (1418) | Sell 0.491 (1693) | Overall 0.466
Momentum_10 : Buy 0.449 (1365) | Sell 0.510 (1741) | Overall 0.483
WILLR       : Buy 0.527 (567) | Sell 0.580 (333) | Overall 0.547
STOCH_D     : Buy 0.495 (503) | Sell 0.550 (211) | Overall 0.511

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.546
  Feature importance: MACD: 0.466 | WILLR: 0.534
Best 3-combo: ('MACD', 'Momentum_10', 'WILLR') -> 0.551
  Feature importance: MACD: 0.392 | Momentum_10: 0.305 | WILLR: 0.303

============================================================
ANALYZING 8H TIMEFRAME (2678 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.525 ( 99) | Sell 0.465 ( 99) | Overall 0.495
Momentum_5  : Buy 0.450 (1075) | Sell 0.522 (1316) | Overall 0.490
Momentum_10 : Buy 0.461 (1014) | Sell 0.526 (1360) | Overall 0.498
WILLR       : Buy 0.507 (440) | Sell 0.590 (205) | Overall 0.533
STOCH_D     : Buy 0.520 (381) | Sell 0.556 (133) | Overall 0.529

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.550
  Feature importance: MACD: 0.321 | WILLR: 0.679
Best 3-combo: ('MACD', 'Momentum_10', 'WILLR') -> 0.540
  Feature importance: MACD: 0.325 | Momentum_10: 0.290 | WILLR: 0.384

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.530 accuracy (1672 signals)
3 timeframes agree: 0.508 accuracy (1202 signals)
Overall collision accuracy: 0.521 (2874 signals)
Collision precision: 0.514
Collision recall: 0.521
Collision rate: 0.537

Analysis completed successfully!

 
Loading 1h data from: ./datasets/Binance_TRXUSDT_1h (1).csv
Loaded 39621 1h records from 2018-06-11 11:00:00 to 2023-10-19 23:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (19821 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.501 (742) | Sell 0.478 (742) | Overall 0.490
Momentum_5  : Buy 0.480 (7923) | Sell 0.472 (7208) | Overall 0.476
Momentum_10 : Buy 0.491 (7797) | Sell 0.479 (7029) | Overall 0.485
WILLR       : Buy 0.555 (2900) | Sell 0.546 (3985) | Overall 0.550
STOCH_D     : Buy 0.552 (2461) | Sell 0.527 (3517) | Overall 0.537

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.545
  Feature importance: MACD: 0.144 | WILLR: 0.856
Best 3-combo: ('Momentum_5', 'Momentum_10', 'WILLR') -> 0.527
  Feature importance: Momentum_5: 0.274 | Momentum_10: 0.196 | WILLR: 0.531

============================================================
ANALYZING 4H TIMEFRAME (9919 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.478 (383) | Sell 0.488 (383) | Overall 0.483
Momentum_5  : Buy 0.490 (4357) | Sell 0.469 (3938) | Overall 0.480
Momentum_10 : Buy 0.485 (4212) | Sell 0.473 (3889) | Overall 0.479
WILLR       : Buy 0.544 (1468) | Sell 0.530 (1895) | Overall 0.536
STOCH_D     : Buy 0.553 (1249) | Sell 0.519 (1637) | Overall 0.534

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.553
  Feature importance: MACD: 0.268 | WILLR: 0.732
Best 3-combo: ('MACD', 'Momentum_10', 'WILLR') -> 0.530
  Feature importance: MACD: 0.296 | Momentum_10: 0.406 | WILLR: 0.298

============================================================
ANALYZING 6H TIMEFRAME (6617 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.440 (250) | Sell 0.456 (250) | Overall 0.448
Momentum_5  : Buy 0.480 (2987) | Sell 0.459 (2699) | Overall 0.470
Momentum_10 : Buy 0.489 (2970) | Sell 0.475 (2709) | Overall 0.482
WILLR       : Buy 0.566 (1003) | Sell 0.536 (1208) | Overall 0.550
STOCH_D     : Buy 0.564 (876) | Sell 0.521 (1067) | Overall 0.540

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('Momentum_5', 'WILLR') -> 0.547
  Feature importance: Momentum_5: 0.350 | WILLR: 0.650
Best 3-combo: ('MACD', 'Momentum_5', 'STOCH_D') -> 0.533
  Feature importance: MACD: 0.261 | Momentum_5: 0.374 | STOCH_D: 0.365

============================================================
ANALYZING 8H TIMEFRAME (4964 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.508 (177) | Sell 0.449 (176) | Overall 0.479
Momentum_5  : Buy 0.507 (2271) | Sell 0.475 (2070) | Overall 0.492
Momentum_10 : Buy 0.503 (2293) | Sell 0.468 (2059) | Overall 0.486
WILLR       : Buy 0.555 (768) | Sell 0.511 (872) | Overall 0.532
STOCH_D     : Buy 0.557 (661) | Sell 0.544 (755) | Overall 0.550

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'STOCH_D') -> 0.576
  Feature importance: MACD: 0.361 | STOCH_D: 0.639
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.521
  Feature importance: MACD: 0.231 | WILLR: 0.375 | STOCH_D: 0.394

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.499 accuracy (3661 signals)
3 timeframes agree: 0.486 accuracy (2036 signals)
Overall collision accuracy: 0.495 (5697 signals)
Collision precision: 0.494
Collision recall: 0.495
Collision rate: 0.574

Analysis completed successfully!

 
Loading 1h data from: ./datasets/ETHUSDT_1h.csv
Loaded 35809 1h records from 2019-12-01 00:00:00 to 2024-01-01 00:00:00
Creating 4h, 6h, and 8h timeframes...

============================================================
ANALYZING 2H TIMEFRAME (17905 records)
============================================================

2h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.475 (636) | Sell 0.443 (636) | Overall 0.459
Momentum_5  : Buy 0.484 (6863) | Sell 0.452 (6181) | Overall 0.469
Momentum_10 : Buy 0.497 (7002) | Sell 0.458 (6168) | Overall 0.478
WILLR       : Buy 0.576 (2199) | Sell 0.542 (3655) | Overall 0.555
STOCH_D     : Buy 0.552 (1717) | Sell 0.524 (3270) | Overall 0.533

2h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('WILLR', 'STOCH_D') -> 0.570
  Feature importance: WILLR: 0.791 | STOCH_D: 0.209
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.559
  Feature importance: MACD: 0.196 | WILLR: 0.587 | STOCH_D: 0.217

============================================================
ANALYZING 4H TIMEFRAME (8953 records)
============================================================

4h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.458 (347) | Sell 0.374 (348) | Overall 0.416
Momentum_5  : Buy 0.495 (3917) | Sell 0.456 (3479) | Overall 0.477
Momentum_10 : Buy 0.503 (3839) | Sell 0.467 (3326) | Overall 0.486
WILLR       : Buy 0.583 (1137) | Sell 0.526 (1998) | Overall 0.547
STOCH_D     : Buy 0.548 (882) | Sell 0.498 (1836) | Overall 0.514

4h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.549
  Feature importance: MACD: 0.303 | WILLR: 0.697
Best 3-combo: ('MACD', 'WILLR', 'STOCH_D') -> 0.564
  Feature importance: MACD: 0.245 | WILLR: 0.510 | STOCH_D: 0.246

============================================================
ANALYZING 6H TIMEFRAME (5969 records)
============================================================

6h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.528 (216) | Sell 0.410 (217) | Overall 0.469
Momentum_5  : Buy 0.494 (2660) | Sell 0.442 (2386) | Overall 0.469
Momentum_10 : Buy 0.510 (2738) | Sell 0.470 (2311) | Overall 0.492
WILLR       : Buy 0.542 (756) | Sell 0.510 (1382) | Overall 0.522
STOCH_D     : Buy 0.547 (600) | Sell 0.507 (1247) | Overall 0.520

6h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('Momentum_5', 'WILLR') -> 0.563
  Feature importance: Momentum_5: 0.594 | WILLR: 0.406
Best 3-combo: ('Momentum_5', 'WILLR', 'STOCH_D') -> 0.534
  Feature importance: Momentum_5: 0.400 | WILLR: 0.301 | STOCH_D: 0.299

============================================================
ANALYZING 8H TIMEFRAME (4477 records)
============================================================

8h Individual Indicator Results:
--------------------------------------------------
MACD        : Buy 0.447 (170) | Sell 0.524 (170) | Overall 0.485
Momentum_5  : Buy 0.516 (2072) | Sell 0.466 (1784) | Overall 0.493
Momentum_10 : Buy 0.520 (2171) | Sell 0.483 (1754) | Overall 0.503
WILLR       : Buy 0.543 (580) | Sell 0.507 (1068) | Overall 0.520
STOCH_D     : Buy 0.546 (463) | Sell 0.513 (969) | Overall 0.524

8h Combination Results (RandomForest):
--------------------------------------------------
Best 2-combo: ('MACD', 'WILLR') -> 0.547
  Feature importance: MACD: 0.554 | WILLR: 0.446
Best 3-combo: ('MACD', 'Momentum_5', 'STOCH_D') -> 0.543
  Feature importance: MACD: 0.449 | Momentum_5: 0.236 | STOCH_D: 0.315

============================================================
CROSS-TIMEFRAME COLLISION ANALYSIS
============================================================

Cross-Timeframe RandomForest Collision Analysis:
--------------------------------------------------
2 timeframes agree: 0.520 accuracy (2575 signals)
3 timeframes agree: 0.501 accuracy (1470 signals)
Overall collision accuracy: 0.513 (4045 signals)
Collision precision: 0.520
Collision recall: 0.513
Collision rate: 0.452
