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

    const firstRow = records[0];
    if (!validator.validateTimestamp(firstRow[timeColumn], timeColumn)) {
      self.postMessage({
        type: "error",
        error: "Invalid timestamp format in first row",
      });
      return;
    }

    const jsonData = records
      .map((row, index) => {
        if (!validator.validateTimestamp(row[timeColumn], timeColumn)) {
          self.postMessage({
            type: "error",
            error: `Invalid timestamp format in row ${index + 2}`,
          });
          return null;
        }

        return {
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
        };
      })
      .filter(Boolean);

    if (jsonData.length) {
      self.postMessage({ type: "complete", data: jsonData });
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
