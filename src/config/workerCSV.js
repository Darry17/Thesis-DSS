import * as Comlink from "comlink";
import { parse } from "csv-parse/browser/esm/sync";

class DataValidator {
  validateTimestamp(timestamp, format) {
    const formats = {
      date: /^\d{4}-\d{2}-\d{2}$/,
      time: /^\d{4}-\d{2}-\d{2}T\d{2}:00:00$/,
      week: /^\d{4}-W\d{2}$/,
    };
    return formats[format].test(timestamp);
  }

  validateRequiredFields(headers) {
    const requiredFields = [
      "solar_power",
      "dhi",
      "dni",
      "ghi",
      "temperature",
      "relative_humidity",
      "solar_zenith_angle",
      "wind_speed",
      "wind_power",
      "dew_point",
    ];
    return requiredFields.every((field) => headers.includes(field));
  }
}

// Linear interpolation function for numerical columns
function interpolateData(data, numericalColumns) {
  const interpolatedData = [...data];
  let interpolationCount = 0; // Track number of interpolated values
  let missingValueCount = 0; // Track number of missing values

  numericalColumns.forEach((column) => {
    let lastValidIndex = null;
    let nextValidIndex = null;

    for (let i = 0; i < interpolatedData.length; i++) {
      let currentValue = interpolatedData[i][column];

      // Normalize: Convert common missing value markers to null
      if (
        currentValue === "" ||
        currentValue === "NaN" ||
        currentValue === "N/A" ||
        currentValue === "-" ||
        currentValue == null ||
        isNaN(Number(currentValue))
      ) {
        currentValue = null;
        interpolatedData[i][column] = null;
        missingValueCount++;
      } else {
        currentValue = Number(currentValue); // Ensure numerical format
        interpolatedData[i][column] = currentValue;
      }

      // Interpolate if the value is null
      if (currentValue == null) {
        // Find the last valid value
        if (lastValidIndex === null) {
          for (let j = i - 1; j >= 0; j--) {
            if (
              interpolatedData[j][column] != null &&
              !isNaN(Number(interpolatedData[j][column]))
            ) {
              lastValidIndex = j;
              break;
            }
          }
        }

        // Find the next valid value
        nextValidIndex = null;
        for (let j = i + 1; j < interpolatedData.length; j++) {
          if (
            interpolatedData[j][column] != null &&
            !isNaN(Number(interpolatedData[j][column]))
          ) {
            nextValidIndex = j;
            break;
          }
        }

        // Perform linear interpolation if both last and next valid values are found
        if (lastValidIndex !== null && nextValidIndex !== null) {
          const lastValue = Number(interpolatedData[lastValidIndex][column]);
          const nextValue = Number(interpolatedData[nextValidIndex][column]);
          const steps = nextValidIndex - lastValidIndex;
          const stepSize = (nextValue - lastValue) / steps;
          const stepsFromLast = i - lastValidIndex;
          interpolatedData[i][column] = lastValue + stepSize * stepsFromLast;
          interpolationCount++;
        } else if (lastValidIndex !== null) {
          // Use last valid value if no next value
          interpolatedData[i][column] = Number(
            interpolatedData[lastValidIndex][column]
          );
          interpolationCount++;
        } else if (nextValidIndex !== null) {
          // Use next valid value if no last value
          interpolatedData[i][column] = Number(
            interpolatedData[nextValidIndex][column]
          );
          interpolationCount++;
        }
      } else {
        // Update lastValidIndex when a valid value is found
        lastValidIndex = i;
      }
    }
  });

  // Log interpolation results for debugging
  self.postMessage({
    type: "debug",
    message: `Found ${missingValueCount} missing values; Interpolated ${interpolationCount} values across ${numericalColumns.length} columns`,
  });

  return interpolatedData;
}

const processFile = async (file) => {
  if (!file) {
    self.postMessage({ type: "error", error: "No file provided" });
    return;
  }

  try {
    const text = await file.text();
    const records = parse(text, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
    });

    if (!records.length) {
      self.postMessage({ type: "error", error: "No data found in CSV file" });
      return;
    }

    const validator = new DataValidator();
    const headers = Object.keys(records[0]);
    const timeColumns = ["date", "time", "week"];
    const timeColumn = timeColumns.find((col) => headers.includes(col));

    if (!timeColumn) {
      self.postMessage({
        type: "error",
        error:
          "CSV must contain exactly one time-related column: date, time, or week.",
      });
      return;
    }

    if (!validator.validateRequiredFields(headers)) {
      self.postMessage({
        type: "error",
        error: "Missing required solar columns",
      });
      return;
    }

    // Validate timestamps
    for (let i = 0; i < records.length; i++) {
      if (!validator.validateTimestamp(records[i][timeColumn], timeColumn)) {
        self.postMessage({
          type: "error",
          error: `Invalid timestamp format in row ${i + 2}`,
        });
        return;
      }
    }

    // Define numerical columns (excluding time column)
    const numericalColumns = [
      "solar_power",
      "dhi",
      "dni",
      "ghi",
      "temperature",
      "relative_humidity",
      "solar_zenith_angle",
      "wind_speed",
      "wind_power",
      "dew_point",
    ];

    // Perform interpolation on numerical columns
    const interpolatedRecords = interpolateData(records, numericalColumns);

    // Convert to JSON format with numerical values
    const jsonData = interpolatedRecords
      .map((row) => ({
        [timeColumn]: row[timeColumn],
        solar_power: Number(row.solar_power),
        dhi: Number(row.dhi),
        dni: Number(row.dni),
        ghi: Number(row.ghi),
        temperature: Number(row.temperature),
        relative_humidity: Number(row.relative_humidity),
        solar_zenith_angle: Number(row.solar_zenith_angle),
        wind_speed: Number(row.wind_speed),
        wind_power: Number(row.wind_power),
        dew_point: Number(row.dew_point),
      }))
      .filter((row) => row !== null && row !== undefined);

    if (jsonData.length) {
      self.postMessage({ type: "complete", data: jsonData });
    } else {
      self.postMessage({
        type: "error",
        error: "No valid data after processing",
      });
    }
  } catch (error) {
    self.postMessage({ type: "error", error: error.message });
  }
};

Comlink.expose({ processFile });

self.onmessage = async (e) => {
  const { file } = e.data;
  await processFile(file);
};
