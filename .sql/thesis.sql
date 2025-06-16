-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 16, 2025 at 02:59 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `thesis`
--

-- --------------------------------------------------------

--
-- Table structure for table `deleted_forecasts`
--

CREATE TABLE `deleted_forecasts` (
  `id` int(11) NOT NULL,
  `forecast_id` int(11) DEFAULT NULL,
  `file_name` varchar(255) NOT NULL,
  `forecast_type` varchar(50) NOT NULL,
  `granularity` varchar(50) NOT NULL,
  `steps` int(11) NOT NULL,
  `model` varchar(50) NOT NULL,
  `date` datetime DEFAULT NULL,
  `deleted_by` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `dhr_forecast`
--

CREATE TABLE `dhr_forecast` (
  `id` int(11) NOT NULL,
  `forecast_id` int(11) DEFAULT NULL,
  `fourier_terms` int(11) NOT NULL,
  `reg_strength` float NOT NULL,
  `ar_order` int(11) NOT NULL,
  `window` int(11) NOT NULL,
  `polyorder` int(11) NOT NULL,
  `updated_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `esn_forecast`
--

CREATE TABLE `esn_forecast` (
  `id` int(11) NOT NULL,
  `forecast_id` int(11) DEFAULT NULL,
  `N_res` int(11) NOT NULL,
  `rho` float NOT NULL,
  `sparsity` float NOT NULL,
  `alpha` float NOT NULL,
  `lambda_reg` float NOT NULL,
  `lags` int(11) NOT NULL,
  `updated_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `forecast`
--

CREATE TABLE `forecast` (
  `id` int(11) NOT NULL,
  `original_filename` varchar(100) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `forecast_type` varchar(50) NOT NULL,
  `granularity` varchar(50) NOT NULL,
  `steps` int(11) NOT NULL,
  `model` varchar(50) NOT NULL,
  `energy_demand` float NOT NULL,
  `max_capacity` float NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `temp_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `history_logs`
--

CREATE TABLE `history_logs` (
  `id` int(11) NOT NULL,
  `forecast_id` int(11) DEFAULT NULL,
  `file_name` varchar(255) NOT NULL,
  `forecast_type` varchar(50) NOT NULL,
  `granularity` varchar(50) NOT NULL,
  `steps` int(11) NOT NULL,
  `model` varchar(50) NOT NULL,
  `date` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `hybrid_forecast`
--

CREATE TABLE `hybrid_forecast` (
  `id` int(11) NOT NULL,
  `forecast_id` int(11) NOT NULL,
  `fourier_terms` int(11) NOT NULL,
  `reg_strength` float NOT NULL,
  `ar_order` int(11) NOT NULL,
  `window` int(11) NOT NULL,
  `polyorder` int(11) NOT NULL,
  `N_res` int(11) NOT NULL,
  `rho` float NOT NULL,
  `sparsity` float NOT NULL,
  `alpha` float NOT NULL,
  `lambda_reg` float NOT NULL,
  `lags` int(11) NOT NULL,
  `n_features` int(11) NOT NULL,
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `temp`
--

CREATE TABLE `temp` (
  `id` int(11) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `created_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `access_control` varchar(20) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `password`, `access_control`, `created_at`) VALUES
(1, 'admin1', '$2b$12$sKlnFTuyLPhj0le0xpMjsu1llZtzoOHmyVLchvP3WW9dZtX9a8Qeu', 'ADMIN', '2025-03-28 06:01:14'),
(2, 'admin2', '$2b$12$6mhFNHp9gpNGksmZ/BCV0efjkvhvcuL4dNa/IGPXtUs/bE6w/5LZm', 'ADMIN', '2025-03-28 06:01:29'),
(3, 'admin3', '$2b$12$Xxl3wTx39bmXbVsDLLS97uiwRFTOxg0fvKf5IyinMq51ZCny8/ej2', 'ADMIN', '2025-03-28 06:01:40');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `deleted_forecasts`
--
ALTER TABLE `deleted_forecasts`
  ADD PRIMARY KEY (`id`),
  ADD KEY `ix_deleted_forecasts_id` (`id`);

--
-- Indexes for table `dhr_forecast`
--
ALTER TABLE `dhr_forecast`
  ADD PRIMARY KEY (`id`),
  ADD KEY `forecast_id` (`forecast_id`);

--
-- Indexes for table `esn_forecast`
--
ALTER TABLE `esn_forecast`
  ADD PRIMARY KEY (`id`),
  ADD KEY `forecast_id` (`forecast_id`);

--
-- Indexes for table `forecast`
--
ALTER TABLE `forecast`
  ADD PRIMARY KEY (`id`),
  ADD KEY `temp_id` (`temp_id`);

--
-- Indexes for table `history_logs`
--
ALTER TABLE `history_logs`
  ADD PRIMARY KEY (`id`),
  ADD KEY `forecast_id` (`forecast_id`);

--
-- Indexes for table `hybrid_forecast`
--
ALTER TABLE `hybrid_forecast`
  ADD PRIMARY KEY (`id`),
  ADD KEY `forecast_id` (`forecast_id`);

--
-- Indexes for table `temp`
--
ALTER TABLE `temp`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `filename` (`filename`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `deleted_forecasts`
--
ALTER TABLE `deleted_forecasts`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `dhr_forecast`
--
ALTER TABLE `dhr_forecast`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `esn_forecast`
--
ALTER TABLE `esn_forecast`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `forecast`
--
ALTER TABLE `forecast`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `history_logs`
--
ALTER TABLE `history_logs`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `hybrid_forecast`
--
ALTER TABLE `hybrid_forecast`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `temp`
--
ALTER TABLE `temp`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `dhr_forecast`
--
ALTER TABLE `dhr_forecast`
  ADD CONSTRAINT `dhr_forecast_ibfk_1` FOREIGN KEY (`forecast_id`) REFERENCES `forecast` (`id`);

--
-- Constraints for table `esn_forecast`
--
ALTER TABLE `esn_forecast`
  ADD CONSTRAINT `esn_forecast_ibfk_1` FOREIGN KEY (`forecast_id`) REFERENCES `forecast` (`id`);

--
-- Constraints for table `forecast`
--
ALTER TABLE `forecast`
  ADD CONSTRAINT `forecast_ibfk_1` FOREIGN KEY (`temp_id`) REFERENCES `temp` (`id`);

--
-- Constraints for table `history_logs`
--
ALTER TABLE `history_logs`
  ADD CONSTRAINT `history_logs_ibfk_1` FOREIGN KEY (`forecast_id`) REFERENCES `forecast` (`id`) ON DELETE SET NULL;

--
-- Constraints for table `hybrid_forecast`
--
ALTER TABLE `hybrid_forecast`
  ADD CONSTRAINT `hybrid_forecast_ibfk_1` FOREIGN KEY (`forecast_id`) REFERENCES `forecast` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
