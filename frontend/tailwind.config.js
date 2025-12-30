/**********************
 * Tailwind configuration
 **********************/
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#0F172A',
        accent: '#22D3EE',
        danger: '#F43F5E',
        warning: '#FACC15',
        success: '#34D399'
      },
      fontFamily: {
        grotesk: ['"Space Grotesk"', 'sans-serif'],
        inter: ['"Inter"', 'sans-serif']
      },
      boxShadow: {
        card: '0 30px 80px -40px rgba(15, 23, 42, 0.45)'
      }
    }
  },
  plugins: []
};
