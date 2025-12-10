/** @type {import("prettier").Config} */
export default {
  // Line width
  printWidth: 100,

  // Indentation
  tabWidth: 2,
  useTabs: false,

  // Quotes and semicolons
  semi: true,
  singleQuote: false,
  jsxSingleQuote: true,
  quoteProps: "as-needed",

  // Trailing commas
  trailingComma: "es5",

  // Spacing
  bracketSpacing: true,
  bracketSameLine: false,
  arrowParens: "always",

  // Line endings
  endOfLine: "lf",

  // Prose wrap
  proseWrap: "preserve",

  // HTML whitespace
  htmlWhitespaceSensitivity: "css",

  // Embedded language formatting
  embeddedLanguageFormatting: "auto",

  // Vue files
  vueIndentScriptAndStyle: false,

  // File-specific overrides
  overrides: [
    {
      files: "*.md",
      options: {
        proseWrap: "always",
        printWidth: 80,
      },
    },
    {
      files: "*.json",
      options: {
        printWidth: 120,
      },
    },
  ],
};
